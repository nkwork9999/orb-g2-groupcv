"""Conservative energy/force/stress chain-rule equivalence for the frozen port."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if mx.default_device() != mx.gpu:
    pytest.skip("Apple GPU required", allow_module_level=True)
pytest.importorskip("orb_models")
from ase import Atoms
from orb_models.forcefield.calculator import ORBCalculator
from orb_g2_cv.mlx_calculator import MLXCalculator
from tests.test_mlx_dense import frozen


@pytest.fixture(scope="module")
def pair(frozen):
    return ORBCalculator(frozen, device="cpu"), MLXCalculator(frozen)


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
def test_general_conservative_outputs_match_unchanged_orb(pair, atoms):
    original, ported = pair
    for calc in pair:
        calc.reset()
    for prop in ["energy", "forces", "stress"]:
        a = original.get_property(prop, atoms)
        b = ported.get_property(prop, atoms)
        np.testing.assert_allclose(
            b, a, atol=1e-4, rtol=1e-5, err_msg=f"{atoms.get_chemical_formula()}:{prop}"
        )
