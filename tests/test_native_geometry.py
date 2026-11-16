"""Compare the exact published ORB features/ZBL and their coordinate VJPs."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")
from ase import Atoms
from tests.test_mlx_dense import frozen
from orb_models.forcefield.atomic_system import ase_atoms_to_atom_graphs
from orb_models.forcefield.nn_util import get_cutoff
from orb_g2_cv.mlx_geometry import geometric_features, ZBL


def test_geometric_features_and_coordinate_vjp_match_frozen_orb(frozen):
    vectors = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [1.3, 0, 0],
            [-1.0, 0.4, 0.5],
            [2.0, -0.3, 1.0],
            [5.99, 0.1, 0.1],
            [6.0, 0, 0],
        ],
        requires_grad=True,
    )
    graph = frozen.model
    r = vectors.norm(dim=-1)
    cutoff = get_cutoff(r)
    basis = graph.rbf_transform(r)
    angular = graph.angular_transform(vectors)
    expected = cutoff * (basis[:, :, None] * angular[:, None, :]).reshape(
        len(vectors), -1
    )
    weights = np.linspace(-0.3, 0.4, expected.numel(), dtype=np.float32).reshape(
        expected.shape
    )
    reference_gradient = torch.autograd.grad(
        (expected * torch.tensor(weights)).sum() + 0.123 * cutoff.sum(), vectors
    )[0]
    frequency = mx.array(graph.rbf_transform.bessel_weights.detach().numpy())
    prefactor = float(graph.rbf_transform.prefactor)

    def loss(v):
        edge, c = geometric_features(v, frequency, prefactor)
        return mx.sum(edge * mx.array(weights)) + 0.123 * mx.sum(c)

    edge, c = geometric_features(
        mx.array(vectors.detach().numpy()), frequency, prefactor
    )
    np.testing.assert_allclose(
        np.asarray(edge), expected.detach().numpy(), atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(c), cutoff.detach().numpy(), atol=1e-6, rtol=1e-5
    )
    actual_gradient = mx.grad(loss)(mx.array(vectors.detach().numpy()))
    np.testing.assert_allclose(
        np.asarray(actual_gradient), reference_gradient.numpy(), atol=1e-4, rtol=1e-5
    )


@pytest.mark.parametrize(
    "atoms",
    [
        Atoms(
            "H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], cell=[50] * 3, pbc=False
        ),
        Atoms("Fe2", positions=[[0, 0, 0], [0.4, 0.1, 0]], cell=[50] * 3, pbc=True),
        Atoms("Si", positions=[[0, 0, 0]], cell=[3] * 3, pbc=True),
        Atoms("H", positions=[[0, 0, 0]], cell=[50] * 3, pbc=False),
    ],
)
def test_zbl_energy_and_vector_gradient_preserve_original_index_convention(
    frozen, atoms
):
    batch = ase_atoms_to_atom_graphs(
        atoms, system_config=frozen.system_config, device="cpu"
    )
    vectors = batch.edge_features["vectors"].detach().requires_grad_(True)
    batch.edge_features["vectors"] = vectors
    expected = frozen.pair_repulsion_fn(batch)["energy"].sum()
    reference_gradient = torch.autograd.grad(expected, vectors)[0]
    numbers = batch.node_features["atomic_numbers_embedding"].argmax(dim=1).numpy()
    port = ZBL(frozen.pair_repulsion_fn)

    def loss(v):
        return port.energy(
            v,
            mx.array(numbers),
            mx.array(batch.senders.numpy()),
            mx.array(batch.receivers.numpy()),
        )

    value, gradient = mx.value_and_grad(loss)(mx.array(vectors.detach().numpy()))
    np.testing.assert_allclose(
        float(value), float(expected.detach()), atol=1e-4, rtol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(gradient), reference_gradient.numpy(), atol=1e-4, rtol=1e-5
    )
