import hashlib

import numpy as np
import pytest
from ase.build import bulk, molecule
from ase.calculators.calculator import PropertyNotImplementedError, all_changes
from ase.calculators.lj import LennardJones
from ase.calculators.singlepoint import SinglePointCalculator

from orb_g2_cv.ase import CalibratedCalculator, load_calculator, verify_file
from orb_g2_cv.potential import RadialHead


def test_multiatom_energy_forces_and_stress_are_sum_of_real_components():
    a = bulk("H", "fcc", a=4.2, cubic=True)
    h = RadialHead(np.ones(36) * 0.001, width=0.8)
    expected = a.copy()
    expected.calc = LennardJones(rc=5)
    correction = h.evaluate(a)
    a.calc = CalibratedCalculator(LennardJones(rc=5), h)
    assert a.get_potential_energy() == pytest.approx(
        expected.get_potential_energy() + correction[0]
    )
    np.testing.assert_allclose(
        a.get_forces(), expected.get_forces() + correction[1], atol=1e-10
    )
    np.testing.assert_allclose(
        a.get_stress(), expected.get_stress() + correction[2], atol=1e-10
    )


def test_nonperiodic_molecule_without_cell_is_supported_without_input_mutation():
    a = molecule("CH4")
    before = a.cell.array.copy()
    a.calc = CalibratedCalculator(
        LennardJones(rc=5), RadialHead(np.zeros(36), width=0.8)
    )
    assert np.isfinite(a.get_potential_energy())
    assert np.isfinite(a.get_forces()).all()
    np.testing.assert_array_equal(a.cell.array, before)


def test_checkpoint_integrity_and_precision_guards_precede_model_import(tmp_path):
    p = tmp_path / "weights"
    p.write_bytes(b"weights")
    assert verify_file(p, hashlib.sha256(b"weights").hexdigest()) == p
    with pytest.raises(ValueError):
        verify_file(p, "0" * 64)
    with pytest.raises(ValueError):
        load_calculator(tmp_path / "absent.json", dtype="float64")
    with pytest.raises(ValueError):
        load_calculator(tmp_path / "absent.json", device="cuda")


@pytest.mark.parametrize("threads", [0, -1, 1.5, "2", True, 8])
def test_invalid_deployment_thread_count_is_rejected_before_checkpoint_io(
    tmp_path, threads
):
    import json

    artifact = {
        "schema_version": 1,
        "base": {"kind": "orb-v3-conservative-inf-mpa", "compile": False},
        "head": {"coefficients": [0.0] * 36, "width": 0.8},
        "runtime": {"torch_threads": threads},
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(artifact))
    with pytest.raises(ValueError):
        load_calculator(p)


@pytest.mark.parametrize("compiled", ["true", 1, None])
def test_invalid_compile_flag_is_rejected_before_checkpoint_io(tmp_path, compiled):
    import json

    artifact = {
        "schema_version": 1,
        "base": {"kind": "orb-v3-conservative-inf-mpa", "compile": compiled},
        "head": {"coefficients": [0.0] * 36, "width": 0.8},
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(artifact))
    with pytest.raises(ValueError):
        load_calculator(p)


def test_mlx_mps_artifact_passes_device_guard_then_verifies_weights(tmp_path):
    import json

    artifact = json.loads(
        (
            __import__("pathlib").Path(__file__).parents[1]
            / "artifacts/orb-g2-groupcv-v1.json"
        ).read_text()
    )
    artifact["runtime"]["backend"] = "mlx"
    p = tmp_path / "metal.json"
    p.write_text(json.dumps(artifact))
    with pytest.raises(FileNotFoundError):
        load_calculator(p, device="mps", base_checkpoint=tmp_path / "missing.ckpt")


@pytest.mark.parametrize("backend", ["cuda", "jax", None, 1])
def test_unknown_runtime_backend_is_rejected_before_weight_io(tmp_path, backend):
    import json

    artifact = json.loads(
        (
            __import__("pathlib").Path(__file__).parents[1]
            / "artifacts/orb-g2-groupcv-v1.json"
        ).read_text()
    )
    artifact["runtime"]["backend"] = backend
    p = tmp_path / "bad-backend.json"
    p.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="backend"):
        load_calculator(p, base_checkpoint=tmp_path / "missing.ckpt")


@pytest.mark.parametrize("geometry", ["cuda-native", "unknown", 1, False])
def test_unknown_geometry_backend_is_rejected_before_weight_io(tmp_path, geometry):
    import json

    artifact = json.loads(
        (
            __import__("pathlib").Path(__file__).parents[1]
            / "artifacts/orb-g2-groupcv-v1.json"
        ).read_text()
    )
    artifact["runtime"]["geometry_backend"] = geometry
    p = tmp_path / "bad-geometry.json"
    p.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="geometry"):
        load_calculator(p, base_checkpoint=tmp_path / "missing.ckpt")


class ObservedLennardJones(LennardJones):
    """Instrument the real potential without replacing its calculations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requests = []
        self.state_checks = 0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        self.requests.append(tuple(properties))
        super().calculate(atoms, properties, system_changes)

    def check_state(self, atoms, tol=1e-15):
        self.state_checks += 1
        return super().check_state(atoms, tol)


@pytest.mark.parametrize("periodic", [False, True])
def test_direct_base_efs_preserves_values_and_calculates_once_per_changed_outer_state(
    periodic,
):
    atoms = bulk("H", "fcc", a=4.2, cubic=True) if periodic else molecule("CH4")
    original_cell = atoms.cell.array.copy()
    original_positions = atoms.positions.copy()
    head = RadialHead(np.linspace(-0.001, 0.001, 36), width=0.8)
    base = ObservedLennardJones(rc=5)
    atoms.calc = CalibratedCalculator(base, head, direct_base_calculation=True)
    for step in range(2):
        expected = atoms.copy()
        if not periodic:
            expected.set_cell([50] * 3)
        expected.calc = LennardJones(rc=5)
        de, df, ds = head.evaluate(atoms)
        assert atoms.get_potential_energy() == pytest.approx(
            expected.get_potential_energy() + de
        )
        np.testing.assert_allclose(atoms.get_forces(), expected.get_forces() + df)
        np.testing.assert_allclose(atoms.get_stress(), expected.get_stress() + ds)
        # Outer repeated-property reads must not re-run the inner potential.
        atoms.get_potential_energy()
        atoms.get_forces()
        assert base.requests == [("energy", "forces", "stress")] * (step + 1)
        assert base.state_checks == 0
        np.testing.assert_array_equal(atoms.cell.array, original_cell)
        if step == 0:
            np.testing.assert_array_equal(atoms.positions, original_positions)
            atoms.positions[1, 0] += 0.1


def test_default_wrapper_keeps_single_point_stale_state_protection():
    atoms = molecule("H2O")
    atoms.set_cell([50] * 3)
    base = SinglePointCalculator(
        atoms, energy=1.25, forces=np.zeros((3, 3)), stress=np.zeros(6)
    )
    atoms.calc = CalibratedCalculator(base, RadialHead(np.zeros(36), width=0.8))
    assert atoms.get_potential_energy() == 1.25
    atoms.positions[1, 0] += 0.1
    with pytest.raises(PropertyNotImplementedError):
        atoms.get_potential_energy()
