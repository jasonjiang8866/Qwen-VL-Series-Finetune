# Wafer distribution map zero-shot benchmark

Independent of `src/train`, classifier heads, LoRA and training scripts. Nothing
imports or changes the training path. Input is **inspection defect coordinates**,
not a wafer surface photograph. No training or few-shot examples are used.

**Status:** CPU tests and a mock pipeline validate software plumbing. No real fab
records, local model weights or CUDA measurements are included. Synthetic scores
are NOT evidence of real wafer classification quality. Default labels are toy
geometry definitions, not an approved manufacturing taxonomy.

## CPU quick start (Python 3.10+)

Run from repository root in a separate virtual environment:

```bash
python -m venv .venv-wafer
source .venv-wafer/bin/activate
pip install -r benchmarks/wafer_zero_shot/requirements-cpu.txt
python -m unittest discover -s benchmarks/wafer_zero_shot/tests -v
python -m benchmarks.wafer_zero_shot synthetic \
  --seed 42 --per-class 4 --output /tmp/wafer-synthetic.jsonl
python -m benchmarks.wafer_zero_shot run --backend mock \
  --data /tmp/wafer-synthetic.jsonl --output /tmp/wafer-mock-results
```

Synthetic generation is seed-repeatable and shuffled. The mock always returns the
first configured label, receives no ground truth and is intentionally not a
geometric classifier. Its score is only a sanity check. Existing files/run
folders are rejected to avoid silently overwriting experiments.

## Local Qwen inference

Install GPU dependencies into the separate environment. Install a PyTorch wheel
appropriate for the host CUDA driver first; `requirements-gpu.txt` constrains the
API range but does not select a CUDA wheel. Record the resulting environment with
`pip freeze` for cross-host replication.

```bash
pip install -r benchmarks/wafer_zero_shot/requirements-gpu.txt
python -m benchmarks.wafer_zero_shot run \
  --backend transformers --model /models/Qwen2.5-VL-3B-Instruct \
  --device cuda:0 --dtype bfloat16 --max-new-tokens 32 --warmup 1 \
  --data /tmp/wafer-synthetic.jsonl --output /tmp/qwen25-wafer-seed42
# Repeat with a local Qwen3-VL Instruct checkpoint and a NEW output directory.
```

Uses stock Transformers 4.57.1 `AutoModelForImageTextToText` and `AutoProcessor`;
accepts `qwen2_5_vl`, `qwen3_vl`, `qwen3_vl_moe` configurations. The full model
must fit on the selected device; no automatic CPU offload or quantization is
performed. CPU inference can use `--device cpu --dtype float32`. BF16 requires
compatible GPU hardware. Instruct models are preferred; reasoning-style output
will be invalid unless it is exactly one configured label.

Loading is local-only with remote code disabled. There are no API calls or data
uploads. Model weights must already exist in the local directory. Use an immutable
checkpoint snapshot path and retain its revision/checksum alongside the report.
Model config, package versions, seed, generation config, dataset hash and git
commit are recorded. Greedy decoding reduces sampling variation but does not
guarantee bit-identical GPU results across hardware, kernels or package versions.

API references:
- https://huggingface.co/docs/transformers/v4.57.1/model_doc/qwen2_5_vl
- https://huggingface.co/docs/transformers/v4.57.1/model_doc/qwen3_vl

## Input and configurable labels

Copy `configs/default.json` and pass `--config /path/config.json` **before** the
`synthetic` or `run` subcommand. `labels` maps exact ASCII identifiers to semantic
definitions. `strategies` contains system templates with `{definitions}` and
`{labels}` placeholders. Escape other literal braces as `{{` and `}}`.
`synthetic_patterns` maps your labels to supported toy generators (`center`,
`ring`, `scratch`, `random`, `empty`); it is only needed for `synthetic`.

Each JSONL record has this schema:

```json
{"id":"example-001","label":"CENTER","coordinates":[[0.1,0.2],[-0.1,0.1]],"radius":1.0,"center":[0,0],"synthetic":true}
```

Use explicit `synthetic:false` for real data, kept only in your authorized local
environment. IDs must be unique. Ground truth must be a configured label.
Coordinates, center and radius share physical units; x points right and y up.
Convert inspection tool conventions before use. Nonfinite or out-of-wafer points
are errors, never silently clipped. Empty coordinate lists are supported.

Rendering is a fixed-size RGB PNG: white field, gray wafer boundary, black fixed
radius dots, no title/legend/class/ID/text. Physical wafer radius controls scale,
not point extents. No orientation augmentation or autoscaling occurs. Duplicates
and overlapping points visually merge; no density correction, notch or die grid
is inferred. Fix size/dot radius across experiments. PNG hashes audit the inputs;
Pillow version is recorded. Dense maps may lose information at raster resolution.

## Experiments and outputs

Every run evaluates the same records, order and images using all three default
strategies: definitions only, geometric feature comparison, contrastive class
comparison. Label order and generation settings stay fixed. These are prompt
ablations, not independent test datasets. Use a separate held-out real set after
choosing prompts; do not tune on final test labels. Include all deployed classes
and expert-adjudicated ambiguous/mixed cases before making deployment claims.
There is no implicit UNKNOWN class: add one with an explicit definition if needed.

- `<strategy>.prompt.json`: exact system/user prompt, with image placeholder.
- `<strategy>.predictions.jsonl`: raw completion, parsed label/null, truth, sample
  ID, latency, synthetic flag and image hash. Contains sensitive data if input does.
- `images/`: rendered maps, indexed by row rather than unsanitized sample IDs.
- `metrics.json`: accuracy, macro-F1, per-class recall/support, confusion matrix,
  invalid output rate, mean/median/p95 latency and peak GPU allocated/reserved bytes.
- `config.json`, `run.json`: effective configuration and provenance.

Parsing strips surrounding whitespace only. Prose, JSON, multiple labels, wrong
case or punctuation are invalid. Only newly generated tokens are decoded so labels
in the input prompt cannot accidentally count as predictions. Invalid outputs
count as errors and false negatives; the confusion matrix has an additional invalid
column. Macro-F1 averages **all configured classes**, assigning zero for undefined
F1/recall (including absent classes); inspect support before comparison.

Batch evaluation is sequential, batch size 1 (not concurrent serving throughput).
Latency includes processor/tokenization, transfer, generation and decoding with
CUDA synchronization; it excludes rendering, model loading and per-strategy
warmup. Peak memory resets after warmup and includes resident model weights, but
not the model-loading peak. CPU/mock memory is `{}`, not a fictitious zero-GPU
measurement. Inference errors fail the run, rather than being mislabeled as invalid
model outputs; earlier prediction rows remain for diagnosis, but a failed run is
not a complete benchmark report. Output folders may contain sensitive maps and
responses; keep them outside git and do not upload enterprise data.

## Extension and validation boundaries

Implement the `Backend` protocol in `backends.py` to add another VLM: `generate`
receives only a PIL image and messages, never the ground truth or sample ID.
Implement synchronization and peak tracking for that backend and register it in
the CLI explicitly. Keep remote adapters opt-in if added in future.

Tests cover deterministic geometry/units/orientation, invalid coordinates, strict
parsing, hand-calculated metrics, synthetic repeatability, the image+text adapter
contract, prompt-token trimming and full mock evaluation. The adapter contract
test uses fake processor/model objects; it does **not** prove a checkpoint loads or
produces correct classifications. Run the local Qwen command on target hardware
and retain outputs before claiming GPU integration or performance validation.
