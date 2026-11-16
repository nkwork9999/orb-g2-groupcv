"""Full geometric/dense vector-energy differentiation on MLX; CPU topology only."""

from typing import ClassVar

import mlx.core as mx
import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from orb_models.forcefield.atomic_system import ase_atoms_to_atom_graphs

from .edge_derivatives import assemble_edge_derivatives
from .mlx_dense import DenseCore
from .mlx_geometry import ZBL, geometric_features


class VectorEnergyCore(DenseCore):
    def __init__(self, model):
        super().__init__(model)
        graph = model.model
        angular = graph.angular_transform
        radial = graph.rbf_transform
        if (
            angular._lmax != 3
            or not angular.normalize
            or angular.normalization != "component"
            or not graph.outer_product_with_cutoff
            or radial.num_bases != 8
            or float(radial.r_max) != 6.0
            or set(graph.node_feature_names) - {"feat"}
            or set(graph.edge_feature_names) - {"feat"}
        ):
            raise ValueError("unsupported frozen ORB geometric configuration")
        head = model.heads[model.energy_name]
        self.frequency = mx.array(radial.bessel_weights.detach().numpy())
        self.prefactor = mx.array(radial.prefactor.detach().numpy())
        self.mu = mx.array(head.normalizer.bn.running_mean.detach().numpy().reshape(()))
        self.sigma = mx.array(
            torch.sqrt(head.normalizer.bn.running_var).detach().numpy().reshape(())
        )
        self.atom_avg = head.atom_avg
        if not model.pair_repulsion:
            raise ValueError("expected original ZBL backbone")
        self.zbl = ZBL(model.pair_repulsion_fn)
        self.arrays.extend(
            [
                self.frequency,
                self.prefactor,
                self.mu,
                self.sigma,
                self.zbl.c,
                self.zbl.d,
                self.zbl.a_exp,
                self.zbl.a_prefactor,
                self.zbl.covalent_radii,
            ]
        )
        mx.eval(self.arrays)
        self.vector_value_gradient = mx.compile(
            mx.value_and_grad(self.raw_energy, argnums=0)
        )
        self.isolated_raw_value = mx.compile(self.isolated_raw_energy)

    def raw_energy(self, vectors, node, senders, receivers, numbers, reference):
        edge, cutoff = geometric_features(vectors, self.frequency, self.prefactor)
        prediction = self.energy(edge, cutoff, node, senders, receivers)
        raw = prediction * self.sigma + self.mu
        if self.atom_avg:
            raw = raw * node.shape[0]
        raw = raw + reference
        return raw + self.zbl.energy(vectors, numbers, senders, receivers)

    def isolated_raw_energy(self, node, reference):
        raw = self.isolated_energy(node) * self.sigma + self.mu
        if self.atom_avg:
            raw = raw * node.shape[0]
        return raw + reference

    def evaluate_vectors(self, vectors, node, senders, receivers, numbers, reference):
        if mx.default_device() != mx.gpu:
            raise ValueError("MLX GPU runtime required")
        if not len(vectors):
            value = self.isolated_raw_value(mx.array(node), mx.array(reference))
            mx.eval(value)
            return float(value.item()), np.zeros_like(vectors)
        value, gradient = self.vector_value_gradient(
            *[
                mx.array(v)
                for v in [vectors, node, senders, receivers, numbers, reference]
            ]
        )
        mx.eval(value, gradient)
        return float(value.item()), np.asarray(gradient)


class NativeCalculator(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.core = VectorEnergyCore(model)
        self._composition_cache = None
        head = model.heads[model.energy_name]
        self._mu = np.float32(head.normalizer.bn.running_mean.detach().numpy()[0])
        self._sigma = np.float32(
            torch.sqrt(head.normalizer.bn.running_var).detach().numpy()[0]
        )
        self._atom_avg = head.atom_avg

    def _composition(self, batch):
        # Frozen atom-number-only input features may be reused for any geometry.
        # Caller-supplied node features bypass the cache completely.
        cacheable = not self.atoms.info.get("node_features")
        key = self.atoms.numbers.tobytes()
        if (
            cacheable
            and self._composition_cache is not None
            and self._composition_cache[0] == key
        ):
            return self._composition_cache[1:]
        node = self.model.model.featurize_nodes(batch).detach().numpy().copy()
        numbers = (
            batch.node_features["atomic_numbers_embedding"].argmax(dim=1).numpy().copy()
        )
        head = self.model.heads[self.model.energy_name]
        reference = np.float32(
            head.reference(batch.atomic_numbers, batch.n_node)
            .detach()
            .numpy()
            .reshape(-1)[0]
        )
        if cacheable:
            self._composition_cache = (key, node, numbers, reference)
        return node, numbers, reference

    def _reported_energy(self, raw, reference, natoms):
        # Match the original predict() normalization round-trip, in float32.
        normalized = np.float32(np.float32(raw) - reference)
        if self._atom_avg:
            normalized = np.float32(normalized / np.float32(natoms))
        normalized = np.float32((normalized - self._mu) / self._sigma)
        value = np.float32(normalized * self._sigma + self._mu)
        if self._atom_avg:
            value = np.float32(value * np.float32(natoms))
        return float(np.float32(value + reference))

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if not len(self.atoms) or not np.isfinite(self.atoms.positions).all():
            raise ValueError("nonempty finite atoms required")
        batch = ase_atoms_to_atom_graphs(
            self.atoms,
            system_config=self.model.system_config,
            device="cpu",
            output_dtype=torch.float32,
            graph_construction_dtype=torch.float32,
        )
        vectors, _, _ = batch.compute_differentiable_edge_vectors(
            use_stress_displacement=False, use_rotation=False
        )
        vectors = vectors.detach().numpy()
        if not np.isfinite(vectors).all() or np.any(
            np.linalg.norm(vectors, axis=1) <= 0
        ):
            raise ValueError("nonfinite or coincident neighbor vectors")
        node, numbers, reference = self._composition(batch)
        senders = batch.senders.numpy()
        receivers = batch.receivers.numpy()
        raw, gradient = self.core.evaluate_vectors(
            vectors, node, senders, receivers, numbers, reference
        )
        forces, stress = assemble_edge_derivatives(
            vectors,
            gradient,
            senders,
            receivers,
            len(self.atoms),
            batch.system_features["cell"][0].numpy(),
        )
        self.results = {
            "energy": self._reported_energy(raw, reference, len(self.atoms)),
            "forces": forces,
            "stress": stress,
        }
        if any(
            not np.isfinite(v).all() for v in map(np.asarray, self.results.values())
        ):
            raise ValueError("nonfinite native MLX prediction")
