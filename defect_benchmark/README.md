# Defect Distribution Map zero-shot benchmark

Independent, inference-only CLI. It never imports or modifies `src/train`, the classification
head, DeepSpeed launchers, or the root dependency environment. Base inspected: `512f424e74f94755d774b6e3786457750677048b`.

## Scope

A Defect Distribution Map plots **inspection defect coordinates**. It is not a
Wafer Bin Map (die grid / electrical pass-fail bins). This benchmark classifies visible
spatial morphology, not yield or physical root cause. No training, demonstrations, adapters,
or production data are included. The five example classes/rules are illustrative and must
be replaced with the agreed inspection taxonomy before a real study. Mixed/ambiguous/empty
maps need explicit rules and, if appropriate, an `other` class in that taxonomy.

## CPU smoke test

From repository root, Python 3.10+:

```bash
python -m pip install Pillow==11.3.0
python -m unittest discover -s defect_benchmark/tests -v
python -m defect_benchmark.run \
  --config defect_benchmark/configs/example.json \
  --manifest defect_benchmark/fixtures/manifest.jsonl \
  --backend mock --output /tmp/ddm-smoke-001
```

The mock returns the first configured class for every image, without seeing labels. Its
metrics test plumbing only: `model_quality_evaluation=false`, all timing/VRAM fields null.
It does not simulate or estimate VLM accuracy. Tests cover parser rejection, invalid-as-error
metrics, absent classes, prefix-overlapping labels, seeded generation, manifest validation,
end-to-end artifacts, and explicit failed-run recording.

Five PNGs and their normalized coordinates are included in `fixtures/` (one per class).
All are generated from a fixed seed with equal initial point counts, identical styling,
neutral sample names and no class names drawn into the images. Regenerate a larger smoke set:

```bash
python -m defect_benchmark.synthetic --output /tmp/ddm-fixtures --seed 42 --per-class 2
```

These idealized patterns are NOT representative validation data and cannot establish a ranking.

## Single-GPU inference environment

Use a separate virtual environment; do not replace the training environment:

```bash
python -m venv .venv-ddm
source .venv-ddm/bin/activate
# Example matched CUDA build; choose a build supported by your GPU and driver.
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r defect_benchmark/requirements.txt
# Only for --bits 4 or --bits 8:
pip install bitsandbytes==0.48.1
```

This is the intended dependency baseline, not a claim of GPU validation. SDPA avoids a
FlashAttention build. Batch size is 1, a single explicit `cuda:N` is used, and no silent CPU
offloading occurs. Start with Qwen2.5-VL-3B-Instruct or Qwen3-VL-4B-Instruct; these are starter
configurations, not a quality ranking or a VRAM fit guarantee. Dense Qwen3-VL only in this MVP;
Thinking and MoE checkpoints are not part of the intended comparison.

```bash
python -m defect_benchmark.run \
  --config defect_benchmark/configs/example.json \
  --manifest defect_benchmark/fixtures/manifest.jsonl \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --output defect_benchmark/outputs/qwen25-smoke

python -m defect_benchmark.run \
  --config defect_benchmark/configs/example.json \
  --manifest defect_benchmark/fixtures/manifest.jsonl \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --output defect_benchmark/outputs/qwen3-smoke
```

For reproducible comparisons, pass `--revision <immutable-HF-commit>` for **each** model;
default `main` is convenient but mutable. Record/retain local model file hashes yourself if
using a local model directory (resolved revision may be null). Use `--local-files-only` with
cached weights to run offline. No image upload/network inference is performed; Hugging Face
may download public model/processor files unless offline mode is requested. `trust_remote_code`
is disabled. Keep production manifests, images, outputs and filesystem paths private.

For Hub models, the config's resolved commit is reused for the weights and processor/tokenizer
loads within a run. This prevents a mutable ref from changing between those loads; it does
not pin separate runs or hash a local model directory.

Defaults: bf16, 16-bit weights, SDPA, min/max image pixel budgets 65,536/262,144,
32 generated tokens, fixed seed 42, one untimed warmup. `--bits 4` (NF4) or `--bits 8`
reduces weight memory but changes the experiment; record separately. `--dtype float16`
is available for devices without bf16. Processor patch/merge rounding can change actual
resolution. Output records include `image_grid_thw` and input token count. Fine dots may
be lost by resizing; inspect preprocessing before using a pixel budget for real evaluation.

## Data and prompt contract

JSON config contains `system_prompt`, ordered `classes: [{id, description}]`, and `rules`.
Class IDs are unique lowercase identifiers. The final system prompt includes the DDM/WBM
separation, all definitions/rules and the exact output requirement. No few-shot examples.

Manifest is JSONL, one image and ground truth per row:

```json
{"id":"wafer_001","image":"images/map001.png","label":"edge_ring"}
```

Paths are relative to the manifest (or absolute local paths). Empty manifests, duplicate IDs,
unknown labels, missing/corrupt images fail before model loading. Labels and sample IDs are
used only for scoring; model inputs contain RGB pixels, the common system prompt and a
constant question. Paths are not inserted into model text. Use label-free source images.

## Output constraints and evaluation

- Default `--output-mode constrained`: token prefix tree permits only configured IDs followed
  by EOS. Greedy decoding is constrained generation, **not** maximum likelihood scoring of
  all complete labels. Label naming/token length can affect decisions. Hold IDs fixed.
- `--output-mode strict`: free greedy generation with the same prompt; accepts only an exact
  ID after whitespace trimming. JSON, prose, case variations and multiple labels are invalid.
- Generation explicitly disables checkpoint-default merging (`use_model_defaults=False`),
  so a checkpoint's sampling defaults cannot override the benchmark's greedy settings.
- Real-backend outputs record `finish_reason` (`eos` or `length`) and `truncated`.
  A response without terminating EOS is invalid even if its text matches a class ID:
  it may be the unfinished prefix of a longer answer. Raw text is retained.
- Constraint mode's low invalid rate is enforced formatting, not evidence of semantic skill.
  Compare models within the same mode; strict mode separately measures instruction adherence.
- Accuracy uses every sample. Invalid outputs count as errors and false negatives for their
  true class. Macro-F1 averages over **all configured classes**, including absent classes with
  zero F1. Report class support and use a fixed class list across runs.
- Confusion matrix has true classes as rows, predicted classes plus `__invalid__` as columns.
- OOM/inference exceptions abort with `status=failed`, preserve completed prediction records,
  and do not produce a misleading aggregate accuracy. Fix the cause and use a new output dir.

Each fresh output directory contains:

| File | Contents |
| --- | --- |
| `run.json` | Config, args, seed, package versions, GPU/CUDA, code commit, model revision, processor/template, manifest and image SHA-256, run status |
| `prompt.txt` | Exact system prompt |
| `predictions.jsonl` | Truth, parsed prediction, raw output, validity, latency, generation time, peak VRAM and token/grid sizes |
| `metrics.json` | Accuracy, macro-F1, invalid rate, supports, confusion matrix and mean/p50/p95/max measurement summaries |

`latency_s` measures image decode + preprocessing + device transfer + generation + output
decode, using CUDA synchronization, excluding model loading, validation/hashing, scoring and
file writes. `generation_s` measures synchronized generate only (including Python constraint
overhead). Warmup is excluded. Percentiles use nearest-rank. VRAM is PyTorch peak allocated
and reserved bytes per measured request, including resident model weights, not total device
usage or incremental KV memory. Reserved memory can retain warmup caches. There is no fake
zero for unavailable measurements. Run models in separate processes on an idle GPU.

Fixed seed and greedy generation improve repeatability but do not guarantee bitwise equality
across GPU architectures, kernels or dependency versions. Freeze the dataset, prompt, class
order, revision, precision and image budget before comparisons. Do not tune prompts on the
held-out evaluation set. For real fab data, separate wafers/lots/time to reduce leakage and
include sparse, mixed and ambiguous cases. Nothing here establishes actual model accuracy.

## Extension interface

`backends.py` exposes `BACKENDS`. An adapter constructor receives runtime args, exposes JSON
serializable `metadata`, and implements `predict(image_path, system_prompt, labels)` returning
`raw_output`, `latency_s`, `generation_s`, `peak_allocated_bytes`, `peak_reserved_bytes`.
Absent measurements must be null. Adapters never receive truth. The runner, parser, metrics,
manifest and provenance remain shared. Add explicit CLI settings when needed; do not silently
fall back to a different model backend or unconstrained mode.

## Validation status / references

CPU tests and mock flow are the available checks. Real Qwen2.5-VL/Qwen3-VL model loading,
quantized inference, actual GPU latency/VRAM and real-data accuracy still require GPU execution.
No model ranking is supplied. The additive GitHub Actions workflow runs CPU checks only.

The separate `tests_cpu` suite uses real Transformers 4.57.1 and PyTorch/torchvision CPU
implementations to check generation-config merging and image processing at patch sizes
14/16, including slow/fast Qwen2 processors. CUDA admission and checkpoint weight loads
remain mocked; missing dependencies fail rather than skip this suite. Run it in a separate
CPU environment with torch 2.8.0, torchvision 0.23.0, transformers 4.57.1 and Pillow 11.3.0:

```bash
HF_HUB_OFFLINE=1 python -m unittest discover -s defect_benchmark/tests_cpu -v
```

See [CODE_REVIEW.md](CODE_REVIEW.md) for confirmed findings, processor smoke evidence,
and the remaining GPU validation boundary.

API references for the pinned inference version:
- https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2_5_vl
- https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen3_vl

## Unified benchmark (PR #1 + PR #2)

`defect_benchmark` is the sole framework. See [INTEGRATION_REVIEW.md](INTEGRATION_REVIEW.md)
for the code comparison, baseline decision and compatibility limitations.

Use `--prompt-strategy definitions|geometry|contrastive`. All strategies share the exact
same ordered class definitions and decision rules; only the reasoning instruction changes.
Run the six controlled cells in separate processes, keeping all other options fixed:

```bash
for strategy in definitions geometry contrastive; do
  for mode in constrained strict; do
    python -m defect_benchmark.run \
      --config defect_benchmark/configs/example.json \
      --manifest defect_benchmark/fixtures/manifest.jsonl \
      --backend mock --prompt-strategy "$strategy" --output-mode "$mode" \
      --output "/tmp/ddm-${strategy}-${mode}"
  done
done
```

Replace `--backend mock` with a pinned model invocation for actual GPU experiments.
The mock does not implement token generation, so these six runs check orchestration only;
CPU adapter tests separately exercise both generation call contracts using fake objects.

One manifest row accepts **exactly one** of `image` or inline `coordinates`:

```json
{"id":"synthetic_001","coordinates":[[0.1,0.2],[-0.2,0.1]],"radius":1.0,"center":[0.0,0.0],"label":"center_cluster","synthetic":true}
```

Coordinates use x-right/y-up with radius and center in matching physical units. Rendering
rejects nonfinite/out-of-wafer points and uses config `render` (size, margin, dot_radius).
Empty coordinates render an empty disk; the example taxonomy has no empty class, so define
one explicitly before evaluating empty maps. No automatic mapping from PR #1's taxonomy.
The synthetic generator also emits `coordinates.jsonl`; equivalent generated coordinate
and image inputs produce the same frozen PNG hashes with the example render settings.
Existing checked-in PNG fixtures preserve their earlier rendering and hashes.

Each run snapshots model inputs as neutral numbered RGB PNGs under `inputs/`, records their
hashes plus source-image hashes or coordinates, and adds an exact prompt SHA-256. Images
are interpreted as stored pixels (EXIF orientation ignored). `synthetic: true|false` is
optional on image/coordinate rows; omission means unknown, not real data. Only mark false
for known real data. Input validation happens before model loading; an invalid input may
leave an incomplete output directory, which should not be treated as a completed run.

Results include per-class precision/recall/F1/support, strategy, output mode, and evidence:

| `evidence_level` | Meaning |
| --- | --- |
| `cpu_mock` | Pipeline check; no real model, latency or VRAM measurement |
| `gpu_synthetic` | Actual GPU backend on explicitly synthetic inputs |
| `gpu_labeled_real_data` | GPU backend on user-declared real labeled inputs; not independent validation |
| `gpu_unspecified_or_mixed_data` | Provenance absent or mixed |

`model_quality_evaluation=true` only indicates a real backend, not representative data.
`real_fab_accuracy_validated` remains false: the CLI cannot certify dataset quality,
independence, taxonomy approval or representative fab accuracy. No GPU or fab validation
was performed as part of this consolidation.
