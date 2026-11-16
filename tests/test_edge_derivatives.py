"""Conservative directed-edge assembly must match upstream strain autograd."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from ase import Atoms
from tests.test_mlx_dense import frozen
from orb_models.forcefield.atomic_system import ase_atoms_to_atom_graphs
from orb_models.forcefield.forcefield_utils import compute_gradient_forces_and_stress
from orb_g2_cv.edge_derivatives import assemble_edge_derivatives


@pytest.mark.parametrize(
    "atoms",
    [
        Atoms(
            "H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], cell=[50] * 3, pbc=False
        ),
        Atoms(
            "Fe2",
            positions=[[0, 0, 0], [2, 1, 0]],
            cell=[[5, 0, 0], [1, 5, 0], [0.5, 1, 5]],
            pbc=True,
        ),
        Atoms("Si", positions=[[0, 0, 0]], cell=[3] * 3, pbc=True),
        Atoms("H", positions=[[0, 0, 0]], cell=[50] * 3, pbc=False),
    ],
)
def test_directed_periodic_edge_vjp_matches_position_and_strain_gradients(
    frozen, atoms
):
    batch = ase_atoms_to_atom_graphs(
        atoms, system_config=frozen.system_config, device="cpu"
    )
    vectors, displacement, generator = batch.compute_differentiable_edge_vectors()
    energy = (vectors.sin() + 0.1 * vectors.square()).sum().reshape(1)
    vector_gradient = torch.autograd.grad(energy, vectors, retain_graph=True)[0]
    forces, stress, _ = compute_gradient_forces_and_stress(
        energy,
        batch.node_features["positions"],
        displacement,
        generator,
        batch.system_features["cell"],
        training=False,
        compute_stress=True,
    )
    actual_forces, actual_stress = assemble_edge_derivatives(
        vectors.detach().numpy(),
        vector_gradient.numpy(),
        batch.senders.numpy(),
        batch.receivers.numpy(),
        len(atoms),
        batch.system_features["cell"][0].numpy(),
    )
    np.testing.assert_allclose(
        actual_forces, forces.detach().numpy(), atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        actual_stress, stress.detach().numpy()[0], atol=1e-5, rtol=1e-5
    )
