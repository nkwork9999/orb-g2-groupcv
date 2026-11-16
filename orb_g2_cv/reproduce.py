"""Reproduce the fixed training state; outer labels are never opened."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .training import train


def load_training_rows(root):
    root=Path(root);artifact=json.loads((root/'artifacts/orb-g2-groupcv-v1.json').read_text())
    payload=(root/'data/train-cache.jsonl').read_bytes()
    if hashlib.sha256(payload).hexdigest()!=artifact['provenance']['train_cache_sha256']:
        raise ValueError('training cache SHA256 differs from frozen input')
    rows=[json.loads(line) for line in payload.splitlines()]
    if not rows or any(r.get('role')!='train' for r in rows):raise ValueError('training rows only')
    if {r['name'] for r in rows}!=set(artifact['selection']['training_ids']):raise ValueError('training IDs differ')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--verify-backbone',action='store_true')
    parser.add_argument('--base-checkpoint',type=Path)
    args=parser.parse_args();root=args.root;rows=load_training_rows(root)
    artifact=json.loads((root/'artifacts/orb-g2-groupcv-v1.json').read_text())
    if args.verify_backbone:
        from ase import Atoms
        from .ase import load_calculator
        calc=load_calculator(root/'artifacts/orb-g2-groupcv-v1.json',base_checkpoint=args.base_checkpoint).base
        atomic={}
        for z in sorted({z for row in rows for z in row['numbers']}):
            a=Atoms(numbers=[z],positions=[[0,0,0]],cell=[50]*3,pbc=False)
            atomic[z]=float(calc.get_potential_energy(a))
        for row in rows:
            atoms=Atoms(numbers=row['numbers'],positions=row['positions'],cell=row['cell'],pbc=row['pbc'])
            expected=sum(atomic[z] for z in row['numbers'])-float(calc.get_potential_energy(atoms))
            np.testing.assert_allclose(expected,row['baseline_atomization_ev'],atol=1e-5,rtol=0)
        print(f'Verified {len(rows)} real backbone cache entries.')
    fitted=train(rows)
    np.testing.assert_allclose(fitted['coefficients'],artifact['head']['coefficients'],rtol=0,atol=1e-10)
    for key in ['ridge','alpha']:
        np.testing.assert_allclose(fitted[key],artifact['selection'][key],rtol=0,atol=1e-12)
    np.testing.assert_allclose(fitted['width'],artifact['head']['width'],rtol=0,atol=1e-12)
    print('Frozen training state reproduced; no holdout/official labels read or artifacts overwritten.')


if __name__=='__main__':main()
