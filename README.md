# ORB-G2 GroupCV

A frozen ORB-v3 conservative MPA backbone with a 36-coefficient smooth, conservative molecular-energy calibration. All selection uses **training-only formula-group cross-validation**. The correction supports multi-atom and periodic systems and supplies analytic forces/stress; it is not limited to isolated diatomics.

The model was trained on **83 multi-atom G2 molecules from 68 formula groups**. A metadata-only split fixed before target export reserves **19 molecules from 18 disjoint formula groups**. Isomers remain together. All one/two-atom systems were excluded from this fine-tuning dataset. Official Matbench Discovery reference labels were not used as fitting targets or hyperparameter objectives.

## Runtime deployment 0.1.1

The trained state is unchanged from source commit `0e192c1` / release v0.1.0. Deployment v0.1.1 uses four CPU threads and exact fast neighbor enumeration for small nonperiodic or provably image-free systems, falling back to the original periodic neighbor list otherwise. Energy/force/stress agreement on 139 training/synthetic geometries is within 2.3e-15 (head only). Training features retain the original implementation.

Runtime selection used only training geometries and no reference labels. The head-only median speedup was 60.4x, not a whole-model speedup. The initial one-thread compilation study failed its prospective 20% latency-reduction threshold and remains rejected. A later mixed-thread timing study was invalidated because thread-pool switching contaminated timings; raw evidence is retained. Corrected fixed-thread blocks selected eager four-thread inference: median 0.06465 seconds versus eager one-thread 0.18982 seconds on the selected training structures. Compilation at two/four threads failed the same-thread 20% threshold and was rejected. These timings are deployment diagnostics, not controlled cross-model leaderboard comparisons. Plans, raw timings, equivalence tests and the deployment freeze are retained in this repository.

## Results available so far

| Evaluation | Unchanged ORB | ORB-G2 GroupCV |
|---|---:|---:|
| Training-only 5-fold grouped CV MAE | 0.5870 eV | 0.3353 eV |
| Fixed held-out formula-macro MAE | 0.4933 eV | 0.2175 eV |
| Fixed held-out sample MAE | 0.4913 eV | 0.2180 eV |

15 of 18 held-out formula groups improved and 3 worsened. The paired formula-group mean error change is -0.2758 eV; the descriptive 10,000-resample group-bootstrap interval is [-0.3871, -0.1699] eV. This small molecular holdout evaluates the present fine-tune; overlap with upstream pretraining has not been proven absent. It does not establish improved bulk, dynamics, or force accuracy.

Targets are **experimental atomization energies extrapolated with zero-point/thermal corrections**, from ASE 3.29.0's merged G2 database. They are **not PBE energy/force labels**. Geometries and reference conventions are preserved; no force or stress targets were fitted.

Official Diatomics predictions will be generated with the unmodified Python3.14 shared runner after freezing this source/candidate. No official leaderboard rank or acceptance is currently claimed.

## Selection protocol and historical exposure

- The plan, eligible metadata and outer split were fixed before training target export. Eligibility is based on 3–14 atoms, atomic numbers at most 17 and zero supplied initial magnetic moments; it does not filter by target or model error.
- Each inner fold excludes entire formulas. Feature scaling and ridge fitting use only that fold's training rows. A fixed 16-trial GP-EI budget (8 initial Sobol, seed42) selects ridge and radial width, with a zero-correction control and fixed shrinkage grid. The final candidate is refit on all 83 training rows and frozen before the first held-out target export.
- The training API rejects held-out rows. Tests alter inner-validation labels and confirm the predictions for that fold do not change. Tests also protect role filtering, immutable split files, finite-difference force/stress consistency, periodic images and multi-atom behavior.
- All outcomes are retained, including regressions and the equal-budget Sobol control. The holdout and official scores never select a replacement checkpoint.
- **Diatomics is test-set-informed.** Earlier PET/ORB Diatomics results and public task rankings influenced backbone choice and redesign. Later freezing does not erase this history. A submission must set `test_set_informed: [diatomics]` and retain this disclosure. This experiment must not be described as a newly unseen official benchmark.
- This replaces the isolated-pair-only scope of the closed [PR #416](https://github.com/janosh/matbench-discovery/pull/416). The lesson from the [earlier Matbench selection review](https://github.com/materialsproject/matbench/pull/363#issuecomment-6379417619) is enforced: never pick feature/model tracks by outer test score or move favorable examples between roles.

## Reproduction

The tested numerical stack uses Python3.14.3, ASE3.29.0, NumPy2.5.4, SciPy1.18.1, scikit-learn1.9.1, Torch2.14.1 and orb-models0.5.4. A uv project must explicitly override `dm-tree==0.1.10` for Python3.14 wheels and `filelock==4.2.10` for the benchmark's core requirement. These documented compatibility overrides do not alter ORB weights. Original environments are preserved.

From this repository, install the `train` extra and reproduce the fixed state from the published training-only cache:

```sh
uv run --extra train python -m orb_g2_cv.reproduce
```

To additionally regenerate and verify every training backbone prediction:

```sh
uv run --extra train --extra runtime python -m orb_g2_cv.reproduce --verify-backbone
```

The upstream 102 MB checkpoint is downloaded separately and checked against SHA256 `bd1840d1488f3cd01d213970d9618a63f971590ad45e5e83bf5404d4b95d216a`. A local path may be passed with `--base-checkpoint`; the hash is still enforced. Reproduction does not read holdout/official targets or overwrite any original artifacts. Original one-shot execution scripts in `scripts/` retain local paths as historical execution records; use the portable module above for reproduction.

Useful records: `plan.json`, `data/split.json`, `data/train-cache.jsonl`, `artifacts/orb-g2-groupcv-v1.json`, `results/training.json`, `results/holdout-evaluation.json`, `results/holdout-paired-summary.json`, and `evidence/pre-fit-freeze.json` / `evaluation-freeze.json`. The holdout values are now published for transparency and must be considered exposed in future development.

## Licensing and attribution

Original new code and correction coefficients: BSD-3-Clause. ORB code/checkpoint: upstream Apache-2.0. The checkpoint is not redistributed here. ASE-derived data retain ASE's notice; we elect LGPL-3.0 under its LGPL-2.1-or-later option for redistributed records. See `licenses/` and original ASE sources. The fixed G2 atomization-energy formula is applied to the merged `g2.data` lookup, avoiding the G1-only lookup issue in its re-exported convenience function; all eligible records are preserved.
