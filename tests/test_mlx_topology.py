"""Guard topology, edge ordering and float32 rounding against frozen ORB."""

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import molecule
from orb_models.forcefield.atomic_system import SystemConfig, ase_atoms_to_atom_graphs
from orb_g2_cv.mlx_topology import prepare_geometry


@pytest.mark.parametrize(
    "atoms",
    [
        molecule("CH4"),
        Atoms(
            "CH",
            positions=[[25, 25, 25], [26.1234567, 25, 25]],
            cell=[50] * 3,
            pbc=True,
        ),
        Atoms("H", positions=[[0, 0, 0]], cell=[50] * 3, pbc=False),
        Atoms("Si", positions=[[0, 0, 0]], cell=[3] * 3, pbc=True),
        Atoms(
            "Fe2",
            positions=[[-0.5, 7, 0.1], [12, -11, 6]],
            cell=[[5, 0, 0], [1, 5, 0], [0.5, 1, 5]],
            pbc=True,
        ),
        Atoms("CH", positions=[[0, 0, 0], [6.000001, 0, 0]], cell=[50] * 3, pbc=True),
        Atoms(
            "H30",
            positions=np.random.default_rng(42).normal(size=(30, 3)),
            cell=[50] * 3,
            pbc=False,
        ),
    ],
)
def test_minimal_geometry_preserves_full_graph_and_recomputed_vectors(atoms):
    config = SystemConfig(radius=6.0, max_num_neighbors=20)
    original = ase_atoms_to_atom_graphs(
        atoms,
        system_config=config,
        device="cpu",
        output_dtype=torch.float32,
        graph_construction_dtype=torch.float32,
    )
    vectors, _, _ = original.compute_differentiable_edge_vectors(
        use_stress_displacement=False, use_rotation=False
    )
    actual = prepare_geometry(atoms, config)
    for x, y in zip(
        actual,
        [
            vectors.detach().numpy(),
            original.senders.numpy(),
            original.receivers.numpy(),
            original.cell[0].numpy(),
        ],
    ):
        np.testing.assert_array_equal(x, y)


def test_minimal_geometry_retains_partial_periodicity_rejection():
    atoms = Atoms("H", positions=[[0, 0, 0]], cell=[10] * 3, pbc=[True, True, False])
    with pytest.raises(NotImplementedError, match="subset"):
        prepare_geometry(atoms, SystemConfig(radius=6.0, max_num_neighbors=20))
