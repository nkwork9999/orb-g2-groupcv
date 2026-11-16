# ORB-G2 GroupCV

A frozen ORB-v3 conservative MPA backbone with a 36-coefficient smooth, conservative molecular-energy calibration. All selection uses **training-only formula-group cross-validation**. The correction supports multi-atom and periodic systems and supplies analytic forces/stress; it is not limited to isolated diatomics.

The model was trained on **83 multi-atom G2 molecules from 68 formula groups**. A metadata-only split fixed before target export reserves **19 molecules from 18 disjoint formula groups**. Isomers remain together. All one/two-atom systems were excluded from this fine-tuning dataset. Official Matbench Discovery reference labels were not used as fitting targets or hyperparameter objectives.

## Runtime deployment 0.1.3

The trained head, model weights, training IDs and selection remain unchanged from v0.1.0. Original ORB spherical harmonics, Bessel features, cutoff and ZBL equations now run alongside the dense network on MLX. A single vector-energy derivative supplies conservative forces and symmetric virial stress. CPU topology construction and original atomic-index conventions are preserved. A single-composition cache stores only fixed input features/reference energy, never coordinates, predictions or labels; custom input features bypass it.

All 83 training and 100 unlabelled synthetic structures passed E/F/stress agreement at atol1e-4 + rtol1e-5 (maximum normalized error 0.36144, threshold 1). The prospective five-repeat paired training-geometry study reduced median latency from 43.35ms to 30.29ms (30.1%). This is a local deployment diagnostic, not a cross-model timing claim. Full source tests: 70 passed; dependency deprecation warnings remain. The factory selects the declared native backend and CPU reproduction stays available. e3nn MIT notices accompany the translated spherical-harmonic equations.

The previous v0.1.2 official sweep completed in 432.31s plus 44.4s load, with all 92 curves finite. Default CDS was 0.79369431, conservative local rank 6/48 against upstream 6d587c1. The prospective native port plan predates those final results. Native v0.1.3 full official evaluation is complete: all92curves finite,81/81eligible vibrational cases scored,106.5s sweep plus16.81s load on Apple M1 Metal. Default CDS0.8274968474 gives a conservative local comparison4/48 against upstream6d587c1. It did not meet the top-three goal; no benchmark PR or website-rank/acceptance claim is made. Historical test exposure remains disclosed. Raw curves and ranking receipt are attached to releasev0.1.3.

## Subsequent rejected runtime experiments

The source and learned state of v0.1.3 remain the selected deployment. Subsequent outcomes are retained separately:

- [CPU adapter overhead](https://github.com/nkwork9999/orb-g2-groupcv/tree/experiments/native-overhead-rejected): all183 numerical cases and81 tests pass, but latency improved only4.66%, below the prospective20% threshold. Not deployed or officially evaluated.
- [Compiled MLX CPU](https://github.com/nkwork9999/orb-g2-groupcv/tree/experiments/mlx-cpu-rejected): three E/F/stress smoke cases and73 tests pass; latency21.79ms versus GPU14.18ms. Full183 numerical gate was not run after timing rejection. Not deployed or officially evaluated.
- [Metal command buffers](https://github.com/nkwork9999/orb-g2-groupcv/tree/experiments/commandbuffer-rejected):15 fresh-process blocks compared limits40/128/256. Default40 was fastest (11.015/11.317/11.916ms). No candidate adopted; unsafe fast synchronization was never enabled.

These are local training-geometry timing studies, not controlled cross-model performance comparisons. The20% threshold is this project's prospective engineering criterion, not a Matbench submission rule. Its failures remain failures; they are not retrospectively relabeled to reach a desired rank. Further speed or hardware evaluation is required before the requested top-three submission can be made.

## Runtime deployment 0.1.2

The trained state is identical to v0.1.0 (83 training molecules, 36 coefficients). MLX0.32.4 performs the dense encoder, attention blocks and energy MLP on the Apple GPU. The original Torch CPU graph, geometric features, reference-energy normalization and pair repulsion remain. MLX feature/cutoff vector-Jacobian products pass through the original Torch geometry to give conservative forces/stress. This is a general graph calculator, with no benchmark-curve table or dimer-only restriction.

All183 training/synthetic geometries passed E/F/stress agreement at atol1e-4 + rtol1e-5; maximum normalized difference0.3825 (threshold1). On the fixed first20 training structures, five alternating paired repetitions gave median CPU0.2564s and MLX0.04403s (5.82x). Other workloads varied on this shared interactive Mac; this is not a controlled cross-model speed claim. It passed the prospective20% reduction gate. No official reference errors enter fitting or backend numerical/timing selection. Earlier score feedback motivated runtime work, so the Diatomics dagger/disclosure remains mandatory.

Negative outcomes remain: v0.1.0's local CDS comparison was6/48; v0.1.1 was7/48 (CDS0.786291, measured587.56s sweep plus2.58s load; other workloads active). Neither met the top-three target and no benchmark PR was submitted for either. Native PyTorch-MPS loading and float64 preprocessing failed; host-graph PyTorch-MPS passed numerical checks but failed the timing gate. MLX initially crashed on empty-edge scatter gradients; tests caught it. Exact isolated-node evaluation and empty zero geometric VJPs fix it without changing learned weights.

The `metal` extra is optional; CPU inference/reproduction remains available. Official Python3.14 macOS14-compatible wheels are pinned by URL because the larger macOS26 extraction exhausted local disk. uv automatically removed failed extraction. Only this task's rejected Torch compiler cache was removed; scientific artifacts, models, data and existing environments were preserved.

Prospective plans, raw paired timings, numerical cases and failed studies are retained. The v0.1.2 full evaluation is complete with the result reported above; no top-three, website-rank or acceptance claim is made. The official sweep/scoring code remains unchanged; the shared CLI device choices and cost-provenance instrumentation add Metal/MLX support. A separate fixed-thread study retained4threads because1/2did not clear its prospective20% improvement gate. The ORB architecture adaptation retains Apache-2.0 attribution/notices alongside BSD-3-Clause calibration code.

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

Official Diatomics predictions use the unchanged Python3.14 sweep/scoring code after freezing each source/candidate. Shared CLI and provenance support are extended for MLX/MPS. No official leaderboard rank or acceptance is currently claimed.

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
