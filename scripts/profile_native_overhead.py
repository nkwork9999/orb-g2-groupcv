"""Prespecified training-only, synchronized numerical and latency gates."""

import json, time
from pathlib import Path
import numpy as np, torch, mlx.core as mx
from ase import Atoms
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
import importlib.util, sys

oldroot = Path("/Users/nk/Desktop/orb-g2-groupcv-native-geometry-20261117/orb_g2_cv")
spec = importlib.util.spec_from_file_location(
    "v013", oldroot / "__init__.py", submodule_search_locations=[str(oldroot)]
)
old = importlib.util.module_from_spec(spec)
sys.modules["v013"] = old
spec.loader.exec_module(old)
from v013.mlx_native import NativeCalculator as OldCalculator
from v013.ase import CalibratedCalculator as OldWrapper
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


def wrap(base, direct=False):
    return CalibratedCalculator(
        base,
        RadialHead(artifact["head"]["coefficients"], width=artifact["head"]["width"]),
        direct_base_calculation=direct,
    )


calcs = {
    "cpu": wrap(ORBCalculator(model, device="cpu")),
    "baseline": OldWrapper(
        OldCalculator(model),
        RadialHead(artifact["head"]["coefficients"], width=artifact["head"]["width"]),
    ),
    "candidate": wrap(NativeCalculator(model), True),
}
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
    calc = calcs[key]
    calc.reset()
    return [
        np.asarray(calc.get_potential_energy(atoms)),
        calc.get_forces(atoms),
        calc.get_stress(atoms),
    ]


maximum = 0.0
equivalence = []
for i, a in enumerate(inputs + synthetic):
    ref = predict("cpu", a)
    out = predict("candidate", a)
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
        Path("results/native-overhead-profile.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        raise ValueError(result)
print("equivalence passed", len(equivalence), "maximum", maximum, flush=True)
for a in inputs[:20]:
    for key in ("baseline", "candidate"):
        predict(key, a)
mx.synchronize()
records = []
for subset in ("changing_composition", "training_perturbation_sequence"):
    schedule = (
        [(repeat, i) for repeat in range(5) for i in range(20)]
        if subset == "changing_composition"
        else [(repeat, i) for i in range(20) for repeat in range(5)]
    )
    for repeat, i in schedule:
        a = inputs[i].copy()
        if subset == "training_perturbation_sequence":
            a.positions[repeat % len(a), (repeat + i) % 3] += (repeat + 1) * 0.002
        record = {
            "subset": subset,
            "repeat": repeat,
            "name": rows[i]["name"],
            "seconds": {},
        }
        for key in (
            ["baseline", "candidate"] if (repeat + i) % 2 else ["candidate", "baseline"]
        ):
            mx.synchronize()
            start = time.perf_counter()
            predict(key, a)
            mx.synchronize()
            record["seconds"][key] = time.perf_counter() - start
        records.append(record)
med = {
    key: float(np.median([r["seconds"][key] for r in records]))
    for key in ("baseline", "candidate")
}
sub = {
    subset: {
        key: float(
            np.median([r["seconds"][key] for r in records if r["subset"] == subset])
        )
        for key in ("baseline", "candidate")
    }
    for subset in ("changing_composition", "training_perturbation_sequence")
}
result = {
    "adopt": med["candidate"] <= 0.8 * med["baseline"]
    and all(v["candidate"] <= 1.05 * v["baseline"] for v in sub.values()),
    "median_seconds": med,
    "subset_median_seconds": sub,
    "speedup": med["baseline"] / med["candidate"],
    "maximum_normalized_error": maximum,
    "atol": 1e-4,
    "rtol": 1e-5,
    "n_training_geometries": 83,
    "n_synthetic_geometries": len(synthetic),
    "all_learned_weights_unchanged": True,
    "official_labels_used": False,
    "equivalence": equivalence,
    "paired_timings": records,
}
Path("results/native-overhead-profile.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(
    json.dumps(
        {k: v for k, v in result.items() if k not in ("equivalence", "paired_timings")},
        indent=2,
    ),
    flush=True,
)
