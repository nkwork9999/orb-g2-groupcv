"""Minimal inputs retaining original ORB topology and vector arithmetic."""

import torch
from orb_models.forcefield import featurization_utilities as feat_util


def prepare_geometry(atoms, system_config):
    if any(atoms.pbc) and not all(atoms.pbc):
        raise NotImplementedError("periodicity along a subset of axes is unsupported")
    with torch.no_grad():
        positions = torch.from_numpy(atoms.positions)
        cell = torch.from_numpy(atoms.cell.array)
        pbc = torch.from_numpy(atoms.pbc)
        if torch.any(cell != 0) and torch.any(pbc):
            positions = feat_util.map_to_pbc_cell(positions, cell)
        edges, _, unit_shifts = feat_util.compute_pbc_radius_graph(
            positions=positions,
            cell=cell,
            pbc=pbc,
            radius=system_config.radius,
            max_number_neighbors=system_config.max_num_neighbors,
            edge_method=None,
            half_supercell=None,
            float_dtype=torch.float32,
            device="cpu",
        )
        # Match the original AtomGraphs.to(float32) then vector recomputation.
        positions = positions.float()
        cell = cell.float()
        shifts = torch.bmm(
            unit_shifts.float().unsqueeze(1),
            cell.unsqueeze(0).expand(len(edges[0]), -1, -1),
        ).squeeze(1)
        vectors = positions[edges[1]] - positions[edges[0]] + shifts
    return vectors.numpy(), edges[0].numpy(), edges[1].numpy(), cell.numpy()
