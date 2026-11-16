"""An explicit CPU backend must perform the same conservative inference."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from ase import Atoms
from orb_models.forcefield.calculator import ORBCalculator
from tests.test_mlx_dense import frozen
from orb_g2_cv.mlx_native import NativeCalculator


@pytest.mark.parametrize(
    "atoms",
    [
        Atoms(
            "H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], cell=[50] * 3, pbc=False
        ),
        Atoms("H", positions=[[0, 0, 0]], cell=[50] * 3, pbc=False),
        Atoms("Si", positions=[[0, 0, 0]], cell=[3] * 3, pbc=True),
    ],
)
def test_explicit_mlx_cpu_matches_original_energy_forces_stress(frozen, atoms):
    previous = mx.default_device()
    try:
        mx.set_default_device(mx.cpu)
        cpu = NativeCalculator(frozen, backend_device="cpu")
        original = ORBCalculator(frozen, device="cpu")
        for prop in ["energy", "forces", "stress"]:
            np.testing.assert_allclose(
                cpu.get_property(prop, atoms),
                original.get_property(prop, atoms),
                atol=1e-4,
                rtol=1e-5,
            )
    finally:
        mx.set_default_device(previous)
