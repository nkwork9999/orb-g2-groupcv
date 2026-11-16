"""Conservative mixed MLX/Torch adapter using the unchanged ORB geometry/VJP."""

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from orb_models.forcefield.atomic_system import ase_atoms_to_atom_graphs
from orb_models.forcefield.forcefield_utils import compute_gradient_forces_and_stress
from orb_models.forcefield.nn_util import get_cutoff
from .mlx_dense import DenseCore


class MLXCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.core = DenseCore(model)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        model = self.model
        batch = ase_atoms_to_atom_graphs(
            self.atoms,
            system_config=model.system_config,
            device="cpu",
            output_dtype=torch.float32,
            graph_construction_dtype=torch.float32,
        )
        vectors, displacement, generator = batch.compute_differentiable_edge_vectors()
        batch.edge_features["vectors"] = vectors
        batch.system_features["stress_displacement"] = displacement
        batch.system_features["generator"] = generator
        graph = model.model
        edge = graph.featurize_edges(batch)
        node = graph.featurize_nodes(batch)
        cutoff = get_cutoff(vectors.norm(dim=-1))
        value, (edge_gradient, cutoff_gradient) = self.core.evaluate(
            edge.detach().numpy(),
            cutoff.detach().numpy(),
            node.detach().numpy(),
            batch.senders.numpy(),
            batch.receivers.numpy(),
        )
        # The proxy has the exact returned value and the feature/cutoff VJPs.
        # Applying the original geometric autograd graph is the ordinary chain rule.
        proxy = (edge * torch.tensor(edge_gradient)).sum() + (
            cutoff * torch.tensor(cutoff_gradient)
        ).sum()
        prediction = proxy - proxy.detach() + torch.tensor(value, dtype=torch.float32)
        head = model.heads[model.energy_name]
        raw_energy = head.denormalize(prediction.reshape(1), batch)
        if model.pair_repulsion:
            raw_energy = raw_energy + model.pair_repulsion_fn(batch)["energy"]
        forces, stress, _ = compute_gradient_forces_and_stress(
            energy=raw_energy,
            positions=batch.node_features["positions"],
            displacement=displacement,
            generator=generator,
            cell=batch.system_features["cell"],
            training=False,
            compute_stress=True,
        )
        # Preserve the upstream predict() normalization round trip too.
        energy = head.denormalize(
            head.normalize(raw_energy, batch, online=False), batch
        )
        self.results = {
            "energy": float(energy.detach().numpy().reshape(-1)[0]),
            "forces": forces.detach().numpy(),
            "stress": stress.detach().numpy().reshape(6),
        }
        if any(
            not np.isfinite(v).all() for v in map(np.asarray, self.results.values())
        ):
            raise ValueError("nonfinite MLX calibrated-backbone prediction")
