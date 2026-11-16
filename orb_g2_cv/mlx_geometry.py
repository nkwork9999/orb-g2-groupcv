"""Exact ORB geometry equations on MLX, with retained upstream attribution.

ORB radial/cutoff/ZBL definitions: Orbital Materials, Apache-2.0.
Spherical-harmonic polynomials: e3nn0.4.4, Mario Geiger et al., MIT.
The frozen model's original atomic-index and normalization conventions are preserved.
"""

import math

import mlx.core as mx


def envelope(radius, maximum, p):
    ratio = radius / maximum
    value = (
        1.0
        - ((p + 1.0) * (p + 2.0) / 2.0) * ratio**p
        + p * (p + 2.0) * ratio ** (p + 1)
        - (p * (p + 1.0) / 2.0) * ratio ** (p + 2)
    )
    return value * (radius < maximum)


def spherical_harmonics3(vectors):
    unit = vectors / mx.maximum(
        mx.sqrt(mx.sum(vectors * vectors, axis=-1, keepdims=True)), 1e-12
    )
    x, y, z = unit[:, 0], unit[:, 1], unit[:, 2]
    y2 = y * y
    x2z2 = x * x + z * z
    a = math.sqrt(3.0) * x * z
    b = math.sqrt(3.0) * x * y
    c = y2 - 0.5 * x2z2
    d = math.sqrt(3.0) * y * z
    e = math.sqrt(3.0) / 2.0 * (z * z - x * x)
    third = [
        math.sqrt(5.0 / 6.0) * (a * z + e * x),
        math.sqrt(5.0) * a * y,
        math.sqrt(3.0 / 8.0) * (4.0 * y2 - x2z2) * x,
        0.5 * y * (2.0 * y2 - 3.0 * x2z2),
        math.sqrt(3.0 / 8.0) * z * (4.0 * y2 - x2z2),
        math.sqrt(5.0) * e * y,
        math.sqrt(5.0 / 6.0) * (e * z - a * x),
    ]
    return mx.stack(
        [
            mx.ones_like(x),
            math.sqrt(3.0) * x,
            math.sqrt(3.0) * y,
            math.sqrt(3.0) * z,
            *[math.sqrt(5.0) * v for v in [a, b, c, d, e]],
            *[math.sqrt(7.0) * v for v in third],
        ],
        axis=-1,
    )


def geometric_features(vectors, frequency, prefactor):
    radius = mx.sqrt(mx.sum(vectors * vectors, axis=-1))
    cutoff = envelope(radius, 6.0, 4)[:, None]
    basis = prefactor * (mx.sin(radius[:, None] * frequency) / radius[:, None])
    angular = spherical_harmonics3(vectors)
    edge = cutoff * (basis[:, :, None] * angular[:, None, :]).reshape(
        vectors.shape[0], 128
    )
    return edge, cutoff


class ZBL:
    def __init__(self, original):
        if int(original.p) != 6 or original.node_aggregation not in ("mean", "sum"):
            raise ValueError("unsupported frozen ZBL configuration")
        self.reduction = original.node_aggregation
        self.c = mx.array(original.c.detach().numpy())
        self.d = mx.array(original.d.detach().numpy())
        self.a_exp = mx.array(original.a_exp.detach().numpy())
        self.a_prefactor = mx.array(original.a_prefactor.detach().numpy())
        self.covalent_radii = mx.array(original.covalent_radii.detach().numpy())

    def energy(self, vectors, numbers, senders, receivers):
        if vectors.shape[0] == 0:
            return mx.array(0.0, dtype=mx.float32)
        # Preserve the published implementation's argmax(one_hot)+1 convention.
        zu = numbers[senders] + 1
        zv = numbers[receivers] + 1
        screening = self.a_prefactor * 0.529 / (zu**self.a_exp + zv**self.a_exp)
        radius = mx.sqrt(mx.sum(vectors * vectors, axis=-1))
        phi = mx.sum(self.c * mx.exp(-self.d * (radius / screening)[None, :]), axis=0)
        coulomb = 14.3996 * zu * zv / radius
        maximum = self.covalent_radii[zu] + self.covalent_radii[zv]
        value = mx.sum(0.5 * coulomb * phi * envelope(radius, maximum, 6))
        return value / len(numbers) if self.reduction == "mean" else value
