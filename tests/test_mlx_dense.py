"""Exact frozen-weight backend port checks, without reference labels."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if mx.default_device() != mx.gpu:
    pytest.skip("Apple GPU required", allow_module_level=True)
torch = pytest.importorskip("torch")
pytest.importorskip("orb_models")
from ase import Atoms
from orb_models.forcefield import pretrained
from orb_models.forcefield.atomic_system import ase_atoms_to_atom_graphs
from orb_models.forcefield.nn_util import get_cutoff
from orb_g2_cv.mlx_dense import DenseCore, convert_module


@pytest.fixture(scope="module")
def frozen():
    torch.set_num_threads(4)
    return pretrained.orb_v3_conservative_inf_mpa(
        weights_path="/Users/nk/Desktop/diatomics-improvement-20261116/checkpoints/orb-v3-conservative-inf-mpa-ranged.ckpt",
        device="cpu",
        precision="float32-highest",
        compile=False,
    )


def test_converter_rejects_unsupported_layer():
    with pytest.raises(ValueError, match="unsupported"):
        convert_module(torch.nn.Conv1d(2, 3, 1))


@pytest.mark.parametrize(
    "atoms",
    [
        Atoms(
            "H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], cell=[50] * 3, pbc=False
        ),
        Atoms("CH", positions=[[0, 0, 0], [1.3, 0, 0]], cell=[50] * 3, pbc=True),
        Atoms("CH", positions=[[0, 0, 0], [8, 0, 0]], cell=[50] * 3, pbc=True),
        Atoms("H", positions=[[0, 0, 0]], cell=[50] * 3, pbc=False),
    ],
)
def test_dense_nodes_energy_and_feature_vjps_match_frozen_torch(frozen, atoms):
    import mlx.core as mx

    core = DenseCore(frozen)
    batch = ase_atoms_to_atom_graphs(
        atoms, system_config=frozen.system_config, device="cpu"
    )
    graph = frozen.model
    node = graph.featurize_nodes(batch).detach()
    edge = graph.featurize_edges(batch).detach().requires_grad_(True)
    cutoff = (
        get_cutoff(batch.edge_features["vectors"].norm(dim=-1))
        .detach()
        .requires_grad_(True)
    )
    nodes, edges = graph._encoder(node, edge)
    for stack in graph.gnn_stacks:
        nodes, edges = stack(nodes, edges, batch.senders, batch.receivers, cutoff)
    expected = frozen.heads[frozen.energy_name](nodes, batch).sum()
    gradients = torch.autograd.grad(expected, (edge, cutoff), allow_unused=True)
    actual_nodes = core.nodes(
        mx.array(node.numpy()),
        mx.array(edge.detach().numpy()),
        mx.array(cutoff.detach().numpy()),
        mx.array(batch.senders.numpy()),
        mx.array(batch.receivers.numpy()),
    )
    np.testing.assert_allclose(
        np.asarray(actual_nodes), nodes.detach().numpy(), atol=1e-4, rtol=1e-5
    )
    value, grads = core.evaluate(
        edge.detach().numpy(),
        cutoff.detach().numpy(),
        node.numpy(),
        batch.senders.numpy(),
        batch.receivers.numpy(),
    )
    np.testing.assert_allclose(value, expected.detach().numpy(), atol=1e-4, rtol=1e-5)
    for actual, reference in zip(grads, gradients):
        if reference is None:
            reference = torch.zeros_like(edge if actual.shape == edge.shape else cutoff)
        np.testing.assert_allclose(
            actual, reference.detach().numpy(), atol=1e-4, rtol=1e-5
        )
