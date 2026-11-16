import json
from pathlib import Path
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from orb_g2_cv.potential import RadialHead
from tests.reference_potential_v010 import RadialHead as ReferenceHead


def test_fast_runtime_matches_frozen_reference_on_training_and_periodic_cases():
    coeff=np.linspace(-.05,.07,36)
    head=RadialHead(coeff,width=1.2);reference=ReferenceHead(coeff,width=1.2)
    rows=[json.loads(l) for l in (Path(__file__).parents[1]/'data/train-cache.jsonl').read_text().splitlines()]
    cases=[Atoms(numbers=r['numbers'],positions=r['positions'],cell=r['cell'],pbc=r['pbc']) for r in rows]
    cases += [bulk('H','fcc',a=3,cubic=True),Atoms('H',positions=[[0,0,0]],cell=[2.1,2.2,2.3],pbc=True),Atoms('HH',positions=[[0,0,0],[49,0,0]],cell=[50]*3,pbc=True)]
    rng=np.random.default_rng(20261116)
    for n in (2,3,8,16):
        positions=rng.uniform(0,5,(n,3))
        cases.extend([Atoms(numbers=rng.integers(1,30,n),positions=positions,cell=[50]*3,pbc=pbc) for pbc in (True,False)])
    for a in cases:
        old=reference.evaluate(a);new=head.evaluate(a)
        for before,after in zip(old,new,strict=True):np.testing.assert_allclose(after,before,atol=1e-10,rtol=1e-10)


def test_small_isolated_system_avoids_slow_neighbor_list(monkeypatch):
    import orb_g2_cv.potential as module
    def forbidden(*args,**kwargs):raise RuntimeError('slow periodic neighbor search called')
    monkeypatch.setattr(module,'neighbor_list',forbidden)
    a=Atoms('CH4',positions=[[0,0,0],[1.1,0,0],[-.4,1,0],[-.4,-.5,.9],[-.4,-.5,-.9]],cell=[50]*3,pbc=True)
    energy,forces,stress=RadialHead(np.ones(36)*.01,width=.8).evaluate(a)
    assert np.isfinite(energy) and abs(energy)>1e-8
    assert np.isfinite(forces).all() and np.isfinite(stress).all()
