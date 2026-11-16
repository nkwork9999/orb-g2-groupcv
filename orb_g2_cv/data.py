"""Metadata-only, immutable grouping and source export boundaries."""
import hashlib
import json
import math
from pathlib import Path


def assign_roles(rows,seed):
    rows=list(rows)
    if len({r['name'] for r in rows})!=len(rows):
        raise ValueError('duplicate sample identity')
    groups=sorted({r['formula'] for r in rows},key=lambda g:hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest())
    if len(groups)<2:
        raise ValueError('at least two formula groups required')
    heldout=set(groups[:math.ceil(.2*len(groups))])
    return [dict(r,role='holdout' if r['formula'] in heldout else 'train') for r in sorted(rows,key=lambda r:r['name'])]


def freeze_manifest(path,rows,plan):
    path=Path(path)
    records=assign_roles(rows,plan['seed'])
    payload={'plan':plan,'records':records}
    with path.open('x') as file:
        json.dump(payload,file,sort_keys=True,indent=2,allow_nan=False)
        file.write('\n')
    return payload


def atomization_energy_ev(name):
    from ase.data import g2
    from ase.symbols import string2symbols
    from ase.units import kcal,mol
    record=g2.data[name]
    value=-record['enthalpy']+record['ZPE']+record['thermal correction']
    for symbol in string2symbols(record['symbols']):
        atom=g2.data[symbol]
        value+=atom['enthalpy']-atom['thermal correction']
    return float(value*kcal/mol)
