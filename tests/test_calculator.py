import hashlib
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule
from ase.calculators.lj import LennardJones
from orb_g2_cv.ase import CalibratedCalculator, load_calculator, verify_file
from orb_g2_cv.potential import RadialHead


def test_multiatom_energy_forces_and_stress_are_sum_of_real_components():
    a=bulk('H','fcc',a=4.2,cubic=True)
    h=RadialHead(np.ones(36)*.001,width=.8)
    expected=a.copy();expected.calc=LennardJones(rc=5)
    correction=h.evaluate(a)
    a.calc=CalibratedCalculator(LennardJones(rc=5),h)
    assert a.get_potential_energy()==pytest.approx(expected.get_potential_energy()+correction[0])
    np.testing.assert_allclose(a.get_forces(),expected.get_forces()+correction[1],atol=1e-10)
    np.testing.assert_allclose(a.get_stress(),expected.get_stress()+correction[2],atol=1e-10)


def test_nonperiodic_molecule_without_cell_is_supported_without_input_mutation():
    a=molecule('CH4');before=a.cell.array.copy()
    a.calc=CalibratedCalculator(LennardJones(rc=5),RadialHead(np.zeros(36),width=.8))
    assert np.isfinite(a.get_potential_energy())
    assert np.isfinite(a.get_forces()).all()
    np.testing.assert_array_equal(a.cell.array,before)


def test_checkpoint_integrity_and_precision_guards_precede_model_import(tmp_path):
    p=tmp_path/'weights';p.write_bytes(b'weights')
    assert verify_file(p,hashlib.sha256(b'weights').hexdigest())==p
    with pytest.raises(ValueError):verify_file(p,'0'*64)
    with pytest.raises(ValueError):load_calculator(tmp_path/'absent.json',dtype='float64')
    with pytest.raises(ValueError):load_calculator(tmp_path/'absent.json',device='cuda')
