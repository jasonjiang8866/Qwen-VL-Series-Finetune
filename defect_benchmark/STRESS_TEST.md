# Synthetic morphology robustness stress test

This opt-in suite probes sensitivity to controlled changes in **inspection defect coordinates**.
It is optional in the existing GPU smoke orchestrator: `synthetic.py`, checked-in fixtures,
the five-image default and all training paths are unchanged. No model ranking, GPU
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

### Image/coordinate content integrity

Generation stores `stress.pixels_sha256` in both manifests, computed from the expected
coordinate rendering. Its exact input is the ASCII header
`ddm-rgb-v1\n{width}\n{height}\n` followed by row-major, 8-bit RGB pixel bytes.
Dimensions are included; PNG compression, chunks, text and orientation metadata are not.
The loader follows the existing RGB conversion without EXIF rotation.

Before backend construction, the loader compares this fingerprint against the decoded
image or freshly rendered coordinates, then freezes that same RGB image for inference.
The observed `pixels_sha256` is retained in run/prediction records alongside existing file
hashes. Coordinate count/hash and suite/config/pair validation still apply. Replacing a PNG,
editing even one RGB pixel, or pairing coordinates with the wrong expected rendering fails.
PNG re-encoding with identical decoded pixels is accepted. This checks preprocessor input;
model-specific resizing/tokenization still occurs inside the unchanged backend.

Earlier PR #3 datasets without `pixels_sha256` are rejected: regenerate them in a new
directory. Do not backfill the expected hash from unverified PNGs. The suite is still
pre-merge and uses the same v1 morphology/seed policy so coordinates remain reproducible.
Manifests are the trusted reference: these fingerprints detect accidental corruption and
mis-pairing, not coordinated rewriting of both data and expected hashes; they are not signatures.

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

### Optional automated GPU comparison matrix

Use a checkout containing this change (stacked on PR #3), rather than PR #2's branch.
Generate the dataset as above, then validate the **complete** experiment plan on CPU:

```bash
python -m defect_benchmark.gpu_smoke --check-only --stress-suite \
  --manifest /tmp/ddm-stress-v1/manifest.jsonl \
  --prompt-strategies definitions geometry contrastive \
  --output /tmp/ddm-stress-plan-001
```

`--stress-suite` and `--manifest` must be supplied together. `coordinates.jsonl` is also
accepted. With neither option, the original eight-cell, five-image smoke is unchanged;
custom manifests cannot silently replace those fixtures. Only explicitly synthetic inputs
are accepted. Every source row is checked for RGB pixel fingerprint, taxonomy/render config,
semantic eligibility and reference pairing **before** CUDA preflight or inference. Old
datasets without fingerprints must be regenerated. No weights, torch import or inference
are required for `--check-only`; `configuration_checked` / `not_executed` is CPU evidence.

The wrapper freezes validated RGB inputs in `fixture_check/` and writes
`selected_manifest.jsonl` pointing to those copies. All children use that same manifest;
each child rechecks pixels/config/pairs. Original source manifest and selected manifest
hashes, selected IDs, labels, seeds, and pair counts are retained in `fixtures.json`.
`plan.json` records exact pinned revisions, hashes and isolated subprocess output paths.
Never edit a frozen run directory; keep the whole root for reproducibility.

For a smaller **future GPU** validation on an existing machine, first check this plan:

```bash
python -m defect_benchmark.gpu_smoke --check-only --stress-suite \
  --manifest /tmp/ddm-stress-v1/manifest.jsonl --stress-replicates 0 \
  --models qwen3 --bits 4 --output-modes strict constrained \
  --output /tmp/ddm-stress-small-plan-001
```

This selects all 68 images of replicate ID 0, including its five anchors and all variants;
it does not take the first N images. IDs are zero-based dataset replicate IDs, distinct
from `--repeats`, which repeats entire model experiments. Unknown/negative replicate IDs
fail. Every source row is validated even when selecting a subset, and reference closure
is checked again after selection. To reduce data further, generate a separate suite with
`--replicates 1`; it retains one complete paired replicate. Do not truncate manifests.

After separately preparing a compatible CUDA environment using [GPU_SMOKE.md](GPU_SMOKE.md),
remove `--check-only` and choose a **new** output root to run either reviewed plan. Nothing
here provisions or starts a paid GPU. The default per-cell timeout is 3,600 seconds,
including model download/load; use `--timeout` when a full stress cell needs longer.
`--local-files-only`, `--repeats`, model/precision/output/prompt filters retain their
existing meanings. Each cell gets a fresh sequential process and model load.

| Plan (one experiment repeat) | Cells | Measured predictions | Excluded warmups |
| --- | ---: | ---: | ---: |
| Replicate 0, Qwen3 NF4, definitions, strict + constrained | 2 | 136 | 2 |
| Full 204 images, both models/precisions, definitions | 8 | 1,632 | 8 |
| Full 204 images, both models/precisions, all three prompts | 24 | 4,896 | 24 |

Each matrix cell retains the existing CUDA/NF4/EOS checks, logs, failure isolation,
latency and VRAM measurements. `summary.json` additionally separates:

- `observed.classification`: labeled-only accuracy/F1/confusion and invalid rate.
- `observed.morphology`: anchors, preserving variants, family/severity/source-class
  breakdowns, paired consistency/coverage and conditional retention. Full pair records
  remain in `<cell>/benchmark/stress_metrics.json`.
- `observed.invalid_output_rates`: all samples, labeled samples and exploratory samples
  separately. Exploratory outputs are retained and never enter accuracy/F1.
- `observed.measurements`: latency/generation time and allocated/reserved memory
  distributions over all measured samples. Model load time remains separate.

The wrapper verifies input/config hashes and prediction provenance, recomputes the
classification, morphology and cost summaries, and checks the exploratory output file.
Missing or inconsistent artifacts fail the cell; correctly rejected invalid strict
outputs remain adherence observations rather than infrastructure failures. Infrastructure
`passed` does not imply good classification or real fab accuracy. CPU mock results,
synthetic GPU inference, and approved labeled fab evaluation are separate evidence levels;
only CPU configuration/mock validation has been performed for this integration.

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
coordinate/config/pixel SHA256, eligibility and rationale. `provenance.json` in the dataset records
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
Tests also reject valid-PNG replacement, single-pixel edits, coordinate/geometry mis-pairing,
missing fingerprints and subpixel coordinate changes before backend construction; PNG
re-encoding and text metadata changes with identical RGB pixels remain valid.
CI generates the full three-replicate suite and runs both manifests through all six
prompt/output cells (12 CPU mock runs). No checkpoint download, enterprise data or CUDA is
needed for those steps.
CI also checks both manifests against all 24 planned GPU cells without starting them.
CPU orchestration tests cover full/small plans, pair-preserving selection, input rejection
before preflight, unchanged defaults, directory isolation, and fabricated result contracts.

Independent CPU reproduction on parent commit `a172efe82cb8e5147941e53f853961fbb73bbb6a`
confirmed that the old loader plus suite validator accepted all three cases: sample 00000's
PNG replaced by sample 00011's valid PNG, one RGB pixel changed to red, and sample 00000's
coordinates replaced by sample 00011's with a recomputed coordinate hash. The new regression
tests reject these inputs while retaining the original five-image smoke behavior.
