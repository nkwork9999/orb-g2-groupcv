"""Frozen ORB dense computation on MLX; geometry remains the original Torch code.

Attention equations follow orb_models.forcefield.gns (Apache-2.0, Orbital Materials).
No weights are fitted, quantized or changed by this adapter.
"""

import numpy as np
import torch
import mlx.core as mx


def convert_module(module, arrays=None):
    """Convert only explicitly supported inference modules; reject everything else."""
    arrays = [] if arrays is None else arrays

    def constant(tensor):
        if tensor.dtype != torch.float32:
            raise ValueError("float32 weights required")
        value = mx.array(tensor.detach().cpu().numpy())
        arrays.append(value)
        return value

    if isinstance(module, torch.nn.Sequential):
        layers = [convert_module(child, arrays) for child in module]

        def sequential(x):
            for layer in layers:
                x = layer(x)
            return x

        return sequential
    if isinstance(module, torch.nn.Linear):
        weight = constant(module.weight)
        bias = constant(module.bias) if module.bias is not None else None
        return lambda x: x @ weight.T + (bias if bias is not None else 0.0)
    if isinstance(module, torch.nn.RMSNorm):
        if len(module.normalized_shape) != 1:
            raise ValueError("unsupported RMS shape")
        weight = constant(module.weight) if module.weight is not None else None
        eps = module.eps if module.eps is not None else np.finfo(np.float32).eps
        return lambda x: mx.fast.rms_norm(x, weight, eps)
    if isinstance(module, torch.nn.SiLU):
        return lambda x: x * mx.sigmoid(x)
    if isinstance(module, torch.nn.Identity):
        return lambda x: x
    if isinstance(module, torch.nn.Dropout) and not module.training:
        return lambda x: x
    raise ValueError(f"unsupported frozen layer: {type(module).__name__}")


class DenseCore:
    """Immutable encoder/attention/energy-MLP port, single general ASE graph."""

    def __init__(self, model):
        if model.training:
            raise ValueError("inference-only frozen model required")
        if mx.default_device() != mx.gpu:
            raise ValueError("MLX GPU runtime required")
        net = model.model
        arrays = []
        self.encode_node = convert_module(net._encoder._node_fn, arrays)
        self.encode_edge = convert_module(net._encoder._edge_fn, arrays)
        self.stacks = []
        for block in net.gnn_stacks:
            if (
                block._attention_gate != "sigmoid"
                or not block._distance_cutoff
                or hasattr(block, "_cond_node_proj")
            ):
                raise ValueError("unsupported ORB attention configuration")
            self.stacks.append(
                tuple(
                    convert_module(m, arrays)
                    for m in [
                        block._node_mlp,
                        block._edge_mlp,
                        block._receive_attn,
                        block._send_attn,
                    ]
                )
            )
        head = model.heads[model.energy_name]
        if head.node_aggregation not in ("mean", "sum"):
            raise ValueError("unsupported energy aggregation")
        self.reduction = head.node_aggregation
        self.readout = convert_module(head.mlp, arrays)
        self.arrays = arrays
        mx.eval(arrays)
        self.value_gradient = mx.compile(mx.value_and_grad(self.energy, argnums=(0, 1)))
        self.isolated_value = mx.compile(self.isolated_energy)

    def nodes(self, node, edge, cutoff, senders, receivers):
        nodes = self.encode_node(node)
        if edge.shape[0] == 0:
            return self.isolated_nodes(node)
        edges = self.encode_edge(edge)
        for node_mlp, edge_mlp, receive, send in self.stacks:
            ra = mx.sigmoid(receive(edges)) * cutoff
            sa = mx.sigmoid(send(edges)) * cutoff
            updated = edge_mlp(
                mx.concatenate([edges, nodes[senders], nodes[receivers]], axis=1)
            )
            received = mx.zeros_like(nodes).at[receivers].add(updated * ra)
            sent = mx.zeros_like(nodes).at[senders].add(updated * sa)
            nodes = nodes + node_mlp(mx.concatenate([nodes, received, sent], axis=1))
            edges = edges + updated
        return nodes

    def isolated_nodes(self, node):
        nodes = self.encode_node(node)
        for node_mlp, _, _, _ in self.stacks:
            zero = mx.zeros_like(nodes)
            nodes = nodes + node_mlp(mx.concatenate([nodes, zero, zero], axis=1))
        return nodes

    def isolated_energy(self, node):
        nodes = self.isolated_nodes(node)
        pooled = (
            mx.mean(nodes, axis=0, keepdims=True)
            if self.reduction == "mean"
            else mx.sum(nodes, axis=0, keepdims=True)
        )
        return self.readout(pooled).sum()

    def energy(self, edge, cutoff, node, senders, receivers):
        nodes = self.nodes(node, edge, cutoff, senders, receivers)
        pooled = (
            mx.mean(nodes, axis=0, keepdims=True)
            if self.reduction == "mean"
            else mx.sum(nodes, axis=0, keepdims=True)
        )
        return self.readout(pooled).sum()

    def evaluate(self, edge, cutoff, node, senders, receivers):
        if mx.default_device() != mx.gpu:
            raise ValueError("MLX GPU runtime required")
        if len(edge) == 0:
            # Empty graphs have no geometric dependence. Avoid MLX's zero-size
            # scatter-gradient compiler path; these exact VJPs are empty zeros.
            value = self.isolated_value(mx.array(node))
            mx.eval(value)
            return float(value.item()), (np.zeros_like(edge), np.zeros_like(cutoff))
        inputs = [mx.array(x) for x in [edge, cutoff, node, senders, receivers]]
        value, gradient = self.value_gradient(*inputs)
        mx.eval(value, gradient)
        return float(value.item()), tuple(np.asarray(x) for x in gradient)
