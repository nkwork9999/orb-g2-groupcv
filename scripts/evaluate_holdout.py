"""One-shot held-out evaluation of the frozen candidate; never feeds training."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from ase import Atoms
from orb_g2_cv.ase import load_calculator
from orb_g2_cv.data import atomization_energy_ev

root=Path(__file__).resolve().parents[1]
freeze=json.loads((root/'evidence/evaluation-freeze.json').read_text())
for name,sha in freeze['sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
assert not (root/'data/holdout-cache.jsonl').exists(),'Holdout already evaluated; preserve its first result'
rows=[r for r in json.loads((root/'data/split.json').read_text())['records'] if r['role']=='holdout']
calc=load_calculator(root/'artifacts/orb-g2-groupcv-v1.json',base_checkpoint='/Users/nk/Desktop/diatomics-improvement-20261116/checkpoints/orb-v3-conservative-inf-mpa-ranged.ckpt')
atomic={}
for z in sorted({z for row in rows for z in row['numbers']}):
    a=Atoms(numbers=[z],positions=[[0,0,0]],cell=[50]*3,pbc=False)
    atomic[z]=float(calc.base.get_potential_energy(a))
result=[];start=time.perf_counter()
for row in rows:
    atoms=Atoms(numbers=row['numbers'],positions=row['positions'],cell=row['cell'],pbc=row['pbc'])
    candidate_energy=calc.get_potential_energy(atoms);candidate_forces=calc.get_forces(atoms)
    baseline_energy=float(calc.base.results['energy'])
    reference=atomization_energy_ev(row['name'])
    baseline_atomization=sum(atomic[z] for z in row['numbers'])-baseline_energy
    predicted_atomization=sum(atomic[z] for z in row['numbers'])-candidate_energy
    assert np.isfinite(candidate_forces).all()
    result.append(row|{'reference_atomization_ev':reference,'baseline_atomization_ev':baseline_atomization,'corrected_atomization_ev':predicted_atomization,'baseline_absolute_error_ev':abs(baseline_atomization-reference),'corrected_absolute_error_ev':abs(predicted_atomization-reference),'candidate_forces_ev_angstrom':candidate_forces.tolist()})
with (root/'data/holdout-cache.jsonl').open('x') as f:
    for row in result:f.write(json.dumps(row,allow_nan=False)+'\n')
summary={}
for tag in ['baseline','corrected']:
    summary[tag+'_formula_macro_mae_ev']=float(np.mean([np.mean([r[tag+'_absolute_error_ev'] for r in result if r['formula']==group]) for group in sorted({r['formula'] for r in result})]))
    summary[tag+'_sample_mae_ev']=float(np.mean([r[tag+'_absolute_error_ev'] for r in result]))
summary|={'n_rows':len(result),'n_formula_groups':len({r['formula'] for r in result}),'seconds':time.perf_counter()-start,'first_evaluation_only':True,'candidate_sha256':freeze['sha256']['artifacts/orb-g2-groupcv-v1.json'],'candidate_reselected_after_holdout':False,'upstream_pretraining_overlap_not_proven_absent':True,'target_scope':'experimental atomization energies extrapolated with ZPE/thermal corrections; not PBE force labels'}
(root/'results/holdout-evaluation.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
