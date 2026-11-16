"""Native geometric graph must preserve original conservative predictions."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from ase import Atoms
from orb_models.forcefield.calculator import ORBCalculator
from tests.test_mlx_dense import frozen
from orb_g2_cv.mlx_native import NativeCalculator


@pytest.fixture(scope="module")
def pair(frozen):
    return ORBCalculator(frozen, device="cpu"), NativeCalculator(frozen)


@pytest.mark.parametrize(
    "atoms",
    [
        Atoms(
            "H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], cell=[50] * 3, pbc=False
        ),
        Atoms("CH", positions=[[0, 0, 0], [1.3, 0, 0]], cell=[50] * 3, pbc=True),
        Atoms("CH", positions=[[0, 0, 0], [8, 0, 0]], cell=[50] * 3, pbc=True),
        Atoms("H", positions=[[0, 0, 0]], cell=[50] * 3, pbc=False),
        Atoms("Si", positions=[[0, 0, 0]], cell=[3] * 3, pbc=True),
        Atoms(
            "Fe2",
            positions=[[0, 0, 0], [2, 1, 0]],
            cell=[[5, 0, 0], [1, 5, 0], [0.5, 1, 5]],
            pbc=True,
        ),
    ],
)
def test_native_full_energy_forces_stress_match_original(pair, atoms):
    original, ported = pair
    for calc in pair:
        calc.reset()
    for prop in ["energy", "forces", "stress"]:
        np.testing.assert_allclose(
            ported.get_property(prop, atoms),
            original.get_property(prop, atoms),
            atol=1e-4,
            rtol=1e-5,
            err_msg=f"{atoms.get_chemical_formula()}:{prop}",
        )


def test_composition_cache_never_reuses_geometry_or_stale_atomic_features(pair):
    import torch

    original, ported = pair
    atoms = Atoms(
        "H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], cell=[50] * 3, pbc=False
    )

    def compare():
        # Force recalculation for the same composition too, so custom info is tested.
        for calc in pair:
            calc.reset()
        values = []
        for prop in ["energy", "forces", "stress"]:
            actual = ported.get_property(prop, atoms)
            expected = original.get_property(prop, atoms)
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-5)
            values.append(np.asarray(actual).copy())
        return values

    first = compare()
    atoms.positions[1, 0] = 1.2
    second = compare()
    assert abs(float(first[0]) - float(second[0])) > 1e-3
    atoms.numbers[2] = 7
    compare()
    atoms.info["node_features"] = {
        "atomic_numbers_embedding": torch.nn.functional.one_hot(
            torch.tensor([2, 1, 7]), 118
        ).float()
    }
    compare()


def test_factory_selects_declared_geometry_and_preserves_calibrated_outputs(tmp_path):
    import json
    from pathlib import Path
    from ase.build import molecule
    from orb_g2_cv.ase import load_calculator

    artifact = json.loads(
        (Path(__file__).parents[1] / "artifacts/orb-g2-groupcv-v1.json").read_text()
    )
    artifact["runtime"]["geometry_backend"] = "mlx-native-float32"
    path = tmp_path / "native.json"
    path.write_text(json.dumps(artifact))
    weights = "/Users/nk/Desktop/diatomics-improvement-20261116/checkpoints/orb-v3-conservative-inf-mpa-ranged.ckpt"
    ported = load_calculator(path, device="mps", base_checkpoint=weights)
    assert isinstance(ported.base, NativeCalculator), (
        "declared native geometry must not silently run hybrid CPU geometry"
    )
    assert ported.mbd_backend == "mlx" and ported.mbd_mlx_arrays
    original = load_calculator(path, device="cpu", base_checkpoint=weights)
    atoms = molecule("CH4")
    for prop in ["energy", "forces", "stress"]:
        np.testing.assert_allclose(
            ported.get_property(prop, atoms),
            original.get_property(prop, atoms),
            atol=1e-4,
            rtol=1e-5,
        )
