"""Conservative position/strain derivatives from general directed periodic edges."""

import numpy as np


def assemble_edge_derivatives(vectors, gradient, senders, receivers, natoms, cell):
    vectors = np.asarray(vectors, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    senders = np.asarray(senders)
    receivers = np.asarray(receivers)
    if (
        vectors.shape != gradient.shape
        or vectors.shape != (len(senders), 3)
        or len(senders) != len(receivers)
    ):
        raise ValueError("inconsistent directed edge derivatives")
    if not np.isfinite(vectors).all() or not np.isfinite(gradient).all():
        raise ValueError("nonfinite edge derivatives")
    forces = np.zeros((natoms, 3))
    np.add.at(forces, senders, gradient)
    np.add.at(forces, receivers, -gradient)
    volume = abs(float(np.linalg.det(np.asarray(cell, dtype=np.float32))))
    if volume == 0:
        stress = np.zeros((3, 3))
    else:
        virial = vectors.T @ gradient
        stress = 0.5 * (virial + virial.T) / volume
        stress = np.where(abs(stress) < 1e10, stress, 0.0)
    return forces, stress[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
