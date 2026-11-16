import numpy as np
import pytest
from ase import Atoms
from orb_g2_cv.potential import RadialHead


def head():return RadialHead(np.linspace(-.03,.04,36),width=.8)


def test_multiatom_force_is_energy_gradient_and_permutation_invariant():
    a=Atoms('CH4',positions=[[0,0,0],[1.1,0,0],[-.4,1,0],[-.4,-.5,.9],[-.4,-.5,-.9]])
    h=head();e,f,_=h.evaluate(a);eps=1e-5
    for i in range(len(a)):
        for j in range(3):
            plus=a.copy();minus=a.copy();plus.positions[i,j]+=eps;minus.positions[i,j]-=eps
            assert f[i,j]==pytest.approx(-(h.evaluate(plus)[0]-h.evaluate(minus)[0])/(2*eps),abs=1e-8)
    order=[3,0,4,2,1];permuted=h.evaluate(a[order])
    assert permuted[0]==pytest.approx(e,abs=1e-12)
    np.testing.assert_allclose(permuted[1],f[order],atol=1e-12)
    np.testing.assert_allclose(f.sum(axis=0),0,atol=1e-12)


def test_periodic_images_stress_matches_cell_energy_derivative():
    a=Atoms('H',positions=[[0,0,0]],cell=[2.1,2.2,2.3],pbc=True)
    h=head();e,f,s=h.evaluate(a)
    assert abs(e)>1e-8
    np.testing.assert_allclose(f,0,atol=1e-12)
    eps=1e-5
    for axis in range(3):
        plus=a.copy();minus=a.copy();cp=a.cell.array.copy();cm=cp.copy();cp[axis]*=1+eps;cm[axis]*=1-eps
        plus.set_cell(cp,scale_atoms=True);minus.set_cell(cm,scale_atoms=True)
        numeric=(h.evaluate(plus)[0]-h.evaluate(minus)[0])/(2*eps*a.get_volume())
        assert s[axis]==pytest.approx(numeric,abs=1e-8)


def test_isolated_atom_unsupported_pairs_and_cutoff_are_exact_zero():
    for a in [Atoms('H'),Atoms('ArH',positions=[[0,0,0],[1,0,0]]),Atoms('HH',positions=[[0,0,0],[5,0,0]])]:
        e,f,s=head().evaluate(a)
        assert e==0
        np.testing.assert_array_equal(f,np.zeros((len(a),3)))
        np.testing.assert_array_equal(s,np.zeros(6))


def test_pair_feature_sum_counts_each_pair_once():
    a=Atoms('HH',positions=[[0,0,0],[1.3,0,0]])
    h=RadialHead(np.ones(36),width=.8)
    r=1.3;x=r/5;envelope=(1-x)**4*(1+4*x)
    # Hand-derived H-H chemistry: 1, 0.1, 0, 0.01.
    expected=1.11*envelope*sum(np.exp(-.5*((r-c)/.8)**2) for c in np.linspace(.6,4.4,9))
    assert h.evaluate(a)[0]==pytest.approx(expected)


@pytest.mark.parametrize('coeff,width',[(np.zeros(35),.8),(np.full(36,np.nan),.8),(np.zeros(36),0)])
def test_invalid_frozen_state_rejected(coeff,width):
    with pytest.raises(ValueError):RadialHead(coeff,width=width)
