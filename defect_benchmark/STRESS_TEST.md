# Synthetic morphology robustness stress test

This opt-in suite probes sensitivity to controlled changes in **inspection defect coordinates**.
It is independent of the five-image GPU smoke: `synthetic.py`, the checked-in fixtures,
`gpu_smoke.py`, its defaults and all training paths are unchanged. No model ranking, GPU
execution or real fab accuracy is established by CPU/mock tests.

## Generate once, reuse across experiments

From repository root, Python 3.10+ and Pillow 11.3.0:

```bash
python -m defect_benchmark.stress --output /tmp/ddm-stress-v1 --seed 42 --replicates 3
python -m defect_benchmark.run \
  --config defect_benchmark/configs/example.json \
  --manifest /tmp/ddm-stress-v1/manifest.jsonl --stress-suite \
  --backend mock --output /tmp/ddm-stress-mock
```

The default is 204 images: 68 per independently seeded replicate, including five original
anchors. It is a compact, fixed severity grid, not a representative distribution or
statistical confidence study. Use more `--replicates` for more procedural realizations;
paired variants are correlated and must not be treated as independent fab wafers.
Existing output directories are refused. Generate a new directory to change seed or version.

`manifest.jsonl` uses rendered PNG paths; `coordinates.jsonl` uses inline normalized
coordinates with explicit unit radius and center. Both use the existing manifest loader
and produce identical model pixels under the example render configuration. All model-facing
filenames are neutral numeric IDs; no class names, parameters, seeds or labels are drawn on
images or sent to the backend. The backend sees the same image/prompt/label-list interface.

The suite freezes the current **example** config fingerprint, including rules and rendering.
`--stress-suite` requires metadata on every row and validates config, reference pairs,
semantics and seeds before loading a backend. It accepts explicit `label: null` only for
exploratory variants. Without this flag, stress manifests and null labels are rejected.
A typo/unknown label is never interpreted as an unlabeled sample. A changed taxonomy/render
config fails closed: update the generator's policy and schema deliberately before relabeling.
Prompt strategy and strict/constrained decoding remain independent CLI options.

## Perturbations and provisional semantic policy

All geometry uses wafer-radius units, x-right/y-up and a fixed wafer outline; no auto-fit.
The original `synthetic.generate` supplies each five-class anchor set and its sparse
background. Random perturbations use their own SHA256-derived seeds, so increasing the
number of replicates does not change earlier samples. Transformations are one-factor
comparisons against anchors, except the explicitly exploratory mixtures.

| Family | Levels per source class / replicate | Scoring policy |
| --- | --- | --- |
| Density | Half the actual points, sampled without replacement; double via one extra bounded ±0.005 x/y jittered point per original | Labeled; arrangement assumed preserved. Densification does not create independent defects and can overlap at pixel resolution |
| Random noise | Add uniform-disk background equal to 10% or 100% of original count | 10% labeled under sparse-background rule; 100% exploratory |
| Rotation | 30°, 90°, 180° about wafer center | Labeled for all five orientation-independent definitions |
| Position offset | Translate x by +0.05 or +0.4 | Only +0.05 for center/localized clusters and scratches, with zero clipped points, is labeled; all other offsets exploratory |
| Sparse | Four sampled points per source class | Exploratory; class may be visually underdetermined |
| Partial ring | 270°, 180°, 90° arcs; random start, 148 ring points + 12 background points | Exploratory; no agreed completeness threshold |
| Mixed morphology | Scratch+ring, center+localized, ring+localized; 25/50/75% first component, 160 total points | Exploratory; includes background inherited from each source |
| Blank | Empty disk, once per replicate | Exploratory; no blank class in example taxonomy |

Translation explicitly discards out-of-disk coordinates and records the discarded count.
Any clipped sample is exploratory, even at the smaller offset. Partial rings are resampled
at fixed count, so they also concentrate angular density; they are **not** point-deletion
occlusions. Mixed components are sampled from each original anchor without replacement;
fractions describe sampled components, not perfectly separated physical defect mechanisms.

These labels are **documented synthetic assumptions**, not user-approved fab annotations.
Visually inspect generated samples before interpreting results: thinning, raster overlap,
and processor resizing can hide intended geometry. Seeds/coordinates make these cases
reproducible, not guaranteed to be semantically unambiguous to humans.
The existing prompt priority `scratch > edge_ring > center_cluster > localized_cluster >
uniform` stays in the model prompt. It is deliberately **not** used as ground truth for
mixed, blank, partial-ring or other ambiguous samples. Such predictions describe model
behavior under that prompt, not whether the model is correct.

## Run a real model only in an already prepared environment

Use the separate CUDA environment and checkpoint pins in [GPU_SMOKE.md](GPU_SMOKE.md).
The following is an example for an existing single GPU; it does not provision a machine:

```bash
CUDA_VISIBLE_DEVICES=0 python -m defect_benchmark.run \
  --config defect_benchmark/configs/example.json \
  --manifest /tmp/ddm-stress-v1/manifest.jsonl --stress-suite \
  --backend transformers --model Qwen/Qwen2.5-VL-3B-Instruct \
  --revision <immutable-HF-commit> --device cuda:0 --dtype bfloat16 --bits 16 \
  --prompt-strategy definitions --output-mode strict \
  --output defect_benchmark/outputs/stress-qwen25-bf16-strict
```

Replace the revision placeholder with the documented pinned commit. Add `--local-files-only`
for cached weights. Run each model/precision/prompt/output-mode combination in a separate
process with a fresh output directory; reuse the **same frozen dataset**. For coordinate
input substitute `coordinates.jsonl`. No inference was executed on a GPU during development.

## Results and denominators

All existing artifacts remain: `run.json`, frozen `inputs/`, prompt, raw predictions, latency,
generation timing and VRAM. Each prediction also retains its stress metadata: source class,
anchor ID, suite/source/perturbation seeds, replicate, family/level, exact parameters, count,
coordinate/config SHA256, eligibility and rationale. `provenance.json` in the dataset records
manifest hashes, generator schema, seed scheme and render settings. Keep it with both
manifests and images. `anchors/` retains the original generator's source files.

| Artifact | Meaning |
| --- | --- |
| `metrics.json` | Existing accuracy/F1/confusion/invalid metrics on **labeled anchors + preserving variants only**; explicit total/labeled/exploratory counts. Timing/VRAM summaries still cover all measured samples |
| `stress_metrics.json` | Separate anchor and preserving-variant metrics, breakdowns by family, severity and source class, exploratory prediction histograms/invalid rates, and full paired results |
| `exploratory_predictions.jsonl` | All null-label samples, raw output, parsed prediction, validity and perturbation metadata; never assigned accuracy |

Pair metrics use only a preserving variant and its original anchor from the **same run**:

- `consistency`: equal valid predictions / **all eligible pairs**. Invalid/invalid is not
  agreement; one invalid answer also contributes zero. Truncated outputs remain invalid
  through the existing parser.
- `valid_pair_coverage`: pairs with both predictions valid / all eligible pairs.
- `consistency_given_both_valid`: equal predictions / pairs with both valid; null when none.
- `retained_accuracy_given_correct_reference`: correctly predicted variants / pairs whose
  anchor prediction was correct; null when there are no correct anchors. This conditions on
  the model's own correct-anchor subset, so it cannot alone fairly rank different models.

Counts are supplied with rates. Each anchor is reused for several pairs. A constant wrong
classifier can score **100% consistency**; inspect labeled accuracy and anchor/variant
breakdowns together. The mock does exactly that and is only a plumbing check. Metrics retain
`evidence_level=cpu_mock`, `model_quality_evaluation=false` and
`real_fab_accuracy_validated=false` for mock runs. Synthetic real-model runs also cannot
validate fab accuracy. Macro-F1 follows the existing rule averaging over all configured
classes, including absent ones; examine support for narrow slices. Empty metric subsets
are null, not fabricated zero accuracy. Inference exceptions retain the existing failed-run
behavior and partial raw predictions, with no successful stress summary.

## CPU validation

```bash
python -m unittest discover -s defect_benchmark/tests -v
```

Tests cover deterministic bytes/seeds, repeat extension, perturbation geometry and coverage,
image/coordinate equivalence, explicit opt-in, broken references/config rejection, ambiguous
label exclusion, invalid-pair denominators, a known three-pair consistency calculation,
constant-wrong predictions, runner failure artifacts and unchanged five-image defaults.
CI also generates the full three-replicate suite and runs the six prompt/output cells with
mock inference. No checkpoint download, enterprise data or CUDA is needed for those steps.
