"""Prespecified training-only, synchronized numerical and latency gates."""

import json, time
from pathlib import Path
import numpy as np, torch, mlx.core as mx
from ase import Atoms
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

from orb_g2_cv.mlx_native import NativeCalculator
from orb_g2_cv.ase import CalibratedCalculator
from orb_g2_cv.potential import RadialHead

torch.set_num_threads(4)
checkpoint = "/Users/nk/Desktop/diatomics-improvement-20261116/checkpoints/orb-v3-conservative-inf-mpa-ranged.ckpt"
start = time.perf_counter()
model = pretrained.orb_v3_conservative_inf_mpa(
    weights_path=checkpoint, device="cpu", precision="float32-highest", compile=False
)
artifact = json.loads(Path("artifacts/orb-g2-groupcv-v1.json").read_text())


def wrap(base):
    return CalibratedCalculator(
        base,
        RadialHead(artifact["head"]["coefficients"], width=artifact["head"]["width"]),
    )


mx.set_default_device(mx.gpu)
calcs = {
    "cpu": wrap(ORBCalculator(model, device="cpu")),
    "gpu": wrap(NativeCalculator(model, backend_device="gpu")),
}
mx.set_default_device(mx.cpu)
calcs["native"] = wrap(NativeCalculator(model, backend_device="cpu"))
load = time.perf_counter() - start
rows = sorted(
    [json.loads(l) for l in Path("data/train-cache.jsonl").read_text().splitlines()],
    key=lambda r: r["name"],
)
inputs = [
    Atoms(numbers=r["numbers"], positions=r["positions"], cell=r["cell"], pbc=r["pbc"])
    for r in rows
]
synthetic = [
    Atoms("H", positions=[[0, 0, 0]], cell=[50] * 3, pbc=False),
    Atoms("Si", positions=[[0, 0, 0]], cell=[3] * 3, pbc=True),
    Atoms(
        "Fe2",
        positions=[[0, 0, 0], [2, 1, 0]],
        cell=[[5, 0, 0], [1, 5, 0], [0.5, 1, 5]],
        pbc=True,
    ),
]
synthetic += [
    Atoms("CH", positions=[[0, 0, 0], [r, 0, 0]], cell=[50] * 3, pbc=True)
    for r in (0.1, 1.3, 5.99, 6.01, 10.0)
]
# Atomic-number coverage without reference labels or official geometries.
synthetic += [
    Atoms(numbers=[z, 6], positions=[[0, 0, 0], [1.7, 0, 0]], cell=[50] * 3, pbc=False)
    for z in range(1, 93)
]


def predict(key, atoms):
    mx.set_default_device(mx.cpu if key == "native" else mx.gpu)
    calc = calcs[key]
    calc.reset()
    return [
        np.asarray(calc.get_potential_energy(atoms)),
        calc.get_forces(atoms),
        calc.get_stress(atoms),
    ]


for a in inputs[:20]:
    for key in ("gpu", "native"):
        predict(key, a)
mx.synchronize()
records = []
for repeat in range(5):
    for i, a in enumerate(inputs[:20]):
        record = {"repeat": repeat, "name": rows[i]["name"], "seconds": {}}
        for key in ["gpu", "native"] if (repeat + i) % 2 else ["native", "gpu"]:
            mx.synchronize()
            start = time.perf_counter()
            predict(key, a)
            mx.synchronize()
            record["seconds"][key] = time.perf_counter() - start
        records.append(record)
med = {
    key: float(np.median([r["seconds"][key] for r in records]))
    for key in ("gpu", "native")
}
timing_adopt = med["native"] <= 0.8 * med["gpu"]
maximum = None
equivalence = []
if timing_adopt:
    maximum = 0.0
    equivalence = []
    for i, a in enumerate(inputs + synthetic):
        ref = predict("cpu", a)
        out = predict("native", a)
        error = max(
            float(np.max(np.abs(x - y) / (1e-4 + 1e-5 * np.abs(y))))
            for x, y in zip(out, ref)
        )
        equivalence.append(
            {
                "index": i,
                "formula": a.get_chemical_formula(),
                "maximum_normalized_error": error,
            }
        )
        maximum = max(maximum, error)
        if error > 1:
            result = {
                "adopt": False,
                "reason": "equivalence failed",
                "maximum_normalized_error": maximum,
                "equivalence": equivalence,
            }
            Path("results/mlx-cpu-profile.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            raise ValueError(result)
    print("equivalence passed", len(equivalence), "maximum", maximum, flush=True)
result = {
    "adopt": med["native"] <= 0.8 * med["gpu"],
    "median_seconds": med,
    "speedup": med["gpu"] / med["native"],
    "maximum_normalized_error": maximum,
    "atol": 1e-4,
    "rtol": 1e-5,
    "load_before_warmup_seconds": load,
    "n_training_geometries": 83 if timing_adopt else 0,
    "n_synthetic_geometries": len(synthetic) if timing_adopt else 0,
    "all_learned_weights_unchanged": True,
    "official_labels_used": False,
    "equivalence": equivalence,
    "paired_timings": records,
}
Path("results/mlx-cpu-profile.json").write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {k: v for k, v in result.items() if k not in ("equivalence", "paired_timings")},
        indent=2,
    ),
    flush=True,
)
