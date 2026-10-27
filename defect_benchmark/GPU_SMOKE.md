# Reproducible single-GPU smoke validation

This entrypoint runs the existing `python -m defect_benchmark.run` CLI in a fresh,
sequential subprocess for each cell. It never provisions a Pod, calls a cloud provider,
uploads inputs, trains a model, or merges a PR. Use an existing Linux x86-64 RTX 5090
32 GB workstation or a user-provisioned RunPod CUDA GPU. No real checkpoint/CUDA/NF4
validation has yet been performed for this entrypoint; CPU tests validate orchestration only.

## Checkout PR #4 and pin the code version

Both the original five-image smoke and the optional 204-image morphology matrix can
use the same PR #4 checkout. From a new parent directory:

```bash
git clone --single-branch --branch feat/gpu-morphology-matrix https://github.com/jasonjiang8866/Qwen-VL-Series-Finetune.git
cd Qwen-VL-Series-Finetune
git fetch origin feat/gpu-morphology-matrix
DDM_COMMIT=f762ec2e45b0e8908af4d966ba038f890ea26455
git checkout --detach "$DDM_COMMIT"
test "$(git rev-parse HEAD)" = "$DDM_COMMIT"
git rev-parse HEAD
```

This full SHA is the reviewed PR #4 implementation baseline. The branch may advance;
fetching it or checking out `FETCH_HEAD` alone does not pin a repeatable experiment.
To use a later reviewed version, replace `DDM_COMMIT` with its **full 40-character
commit SHA**, fetch the branch, and repeat the detach/assert steps. In an existing
clone, run from its root starting at `git fetch`; keep local changes separate and
retain the recorded SHA with results and the package inventory. PR #2's
`feat/defect-zero-shot-benchmark` branch does not contain the GPU stress-matrix options.
No PR merge is required to use this checkout.

With no `--stress-suite` or `--manifest`, the default remains **8 cells × 5 checked-in
images**. The optional matrix requires both flags and a separately generated suite;
its default dataset has **204 images**, not five. See
[STRESS_TEST.md](STRESS_TEST.md#optional-automated-gpu-comparison-matrix) for full and
pair-preserving small CPU plans.

## Environment on a workstation or existing Pod

After the checkout above, use Python 3.12 and an independent virtual environment.
On RunPod, put the repository, Hugging Face cache and results on your mounted
persistent volume (example: `/workspace`). For CPU-only plans, skip these CUDA
installation steps and use [CPU configuration check](#cpu-configuration-check).

```bash
python3.12 -m venv .venv-ddm
source .venv-ddm/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r defect_benchmark/requirements-gpu-smoke.txt
python -m pip check
nvidia-smi
```

The driver must support the cu128 wheel and the GPU. A template's installed
CUDA toolkit alone is not sufficient: verify the runtime with the preflight below.
PyTorch's CUDA 12.8 wheels support Blackwell; bitsandbytes 0.48.1 supplies CUDA 12.8+
builds with SM120 support on Linux x86-64. These are dependency support statements,
not evidence that either checkpoint has run successfully here. SDPA is used; do not
install FlashAttention, DeepSpeed or the root training requirements for this smoke.

Official baseline references:
- https://pytorch.org/get-started/previous-versions/#v280
- https://discuss.pytorch.org/t/pytorch-support-for-nvidia-rtx-5090/221612
- https://huggingface.co/docs/bitsandbytes/v0.48.1/installation

## CPU configuration check

After the pinned checkout above, use an independent Python environment with only
Pillow for this path; no torch import, checkpoint download or inference:

```bash
python -m pip install Pillow==11.3.0
python -m unittest discover -s defect_benchmark/tests -v
python -m defect_benchmark.gpu_smoke --check-only --output /tmp/ddm-plan-001
```

It checks CLI argument compatibility, immutable model revisions, unique cell paths,
and the checked-in synthetic manifest/images; it writes a plan and frozen fixture
hashes. `configuration_checked` / `not_executed` must not be reported as GPU success.
The existing GitHub CPU workflow runs these checks, with no GPU allocation.

For the optional morphology matrix, retain the PR #4 checkout above (stacked on
PR #3) and pass **both** `--stress-suite --manifest <stress-manifest.jsonl>`.
See [STRESS_TEST.md](STRESS_TEST.md#optional-automated-gpu-comparison-matrix) for full and
pair-preserving small plans, CPU-only validation and separate robustness summaries.
The default remains the five checked-in images. A custom manifest without explicit
stress opt-in is rejected; the wrapper accepts only explicitly synthetic stress data.

## Run the eight-cell GPU matrix

```bash
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/workspace/hf-cache
export HF_HUB_DISABLE_TELEMETRY=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m defect_benchmark.gpu_smoke --output /workspace/ddm-results/smoke-001
```

Choose exactly one physical GPU with `CUDA_VISIBLE_DEVICES`; every child uses its
logical `cuda:0`. Stop unrelated GPU workloads first. Preflight requires one visible
CUDA device, bf16 support and a successful bf16 CUDA matmul; Blackwell rejects
PyTorch runtimes below CUDA 12.8. It does not silently fall back to CPU or fp16.
No fixed VRAM-fit guarantee is made. Keep free disk space for both checkpoint
snapshots, dependencies and outputs (allow roughly 40 GB to start, plus headroom).
The first run downloads public model weights into `HF_HOME`; downloading does not
upload input images. For interrupted downloads, retry with a **new result directory**;
the HF cache can be reused.

| Model key | Public checkpoint | Immutable revision |
| --- | --- | --- |
| `qwen25` | `Qwen/Qwen2.5-VL-3B-Instruct` | `66285546d2b821cf421d4f5eb2576359d3770cd3` |
| `qwen3` | `Qwen/Qwen3-VL-4B-Instruct` | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |

Each model runs 16-bit bf16 and 4-bit NF4 (bf16 compute, double quantization), each
with strict and constrained generation. Defaults: `definitions` prompt, seed 42,
five checked-in synthetic maps, batch size 1, SDPA, 65,536–262,144 pixel budget,
32 output tokens and one excluded warmup. Each cell loads its own model; CUDA
memory is released at process exit. Loading/download time is not inference latency.

Useful repeat/subset/offline options (always choose a fresh output root):

```bash
# All three prompt strategies, two repeats: 48 independent subprocesses.
python -m defect_benchmark.gpu_smoke --prompt-strategies definitions geometry contrastive --repeats 2 --output /workspace/ddm-results/all-prompts-001
# Debug one cell after inspecting a failure.
python -m defect_benchmark.gpu_smoke --models qwen3 --bits 4 --output-modes constrained --output /workspace/ddm-results/qwen3-nf4-002
# Only after BOTH full model snapshots have been cached successfully.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m defect_benchmark.gpu_smoke --local-files-only --output /workspace/ddm-results/offline-001
```

Each cell has a 3,600-second timeout including download and load. Increase `--timeout`
for slow initial downloads. Nonzero exit, OOM, timeout or malformed/missing artifacts
fail that cell, but the remaining matrix continues. Exit code is 1 for any failed
cell/preflight, 130 on interruption, 0 for a passed matrix or a successful check-only
run. Read `summary.json` to distinguish those two exit-0 cases. A run never overwrites
or resumes an existing output directory. Run the same plan under a new root to repeat.

## Evidence and pass criteria

The runner verifies the exact revision and requested settings, all parameters on
`cuda:0`, actual bf16 parameter presence, and observed bitsandbytes `Linear4bit`
modules with bf16 compute for NF4 cells. Completed CLI status is the checkpoint-load
and forward/generate evidence; a requested `--bits 4` flag alone is insufficient.

Generated token IDs and tokenizer EOS ID are retained alongside decoded raw text.
Checks compare the last token with EOS, token count/budget, finish reason, truncation
flag and parsed-label validity. No tokens may follow EOS. A length-limited response
must be invalid even if its decoded text equals a class. Constrained cells must
produce a configured label followed by EOS for every sample.

Strict mode can legitimately produce prose, an invalid label or reach the token
limit. Correctly recorded/rejected strict outputs do not fail the **infrastructure**
gate: `strict_adherence_pass`, `all_outputs_eos`, `length_count`, `invalid_count`
and warnings report this separately. Thus `status=passed` alone does **not** mean
every strict response ended with EOS or followed the prompt. Inspect those fields.
For a deliberate truncation probe after the normal matrix, run a strict subset with
`--max-new-tokens 1` in a new root; non-EOS answers must be invalid. If the model
immediately emits EOS, the negative truncation case was not exercised. Do not use a
one-token constrained run: the CLI requires room for the longest label plus EOS.

Per-sample latency, generation time and allocated/reserved GPU peaks must be positive
finite values. Timings exclude warmup/model load; VRAM includes resident weights,
and reserved memory may retain warmup allocations. `model_load_and_processor_s`
includes loading and processor setup (and cache misses/downloads), recorded separately.
These five synthetic samples are too few for performance conclusions or model rankings.
There is no synthetic accuracy threshold. Real fab accuracy remains unvalidated.

## Retained artifacts

| Location | Evidence |
| --- | --- |
| `plan.json` | Full immutable matrix and exact CLI argv |
| `environment.json` | Python/package versions, git SHA/dirty paths, driver/GPU inventory, selected non-secret runtime settings |
| `fixtures.json`, `fixture_check/` | Synthetic-only provenance, input/config hashes, frozen input pixels |
| `preflight.json` | Actual CUDA kernel admission, device capability/runtime; absent on failed preflight/check-only |
| `summary.json` | Incremental per-cell status, errors, EOS/adherence counts and latency/VRAM summaries |
| `<cell>/command.json`, `status.json`, `stdout.log`, `stderr.log` | Exact command, exit/timeout/wall time, raw logs including load failures |
| `<cell>/benchmark/` | Existing CLI `run.json`, raw `predictions.jsonl`, prompt, frozen inputs and `metrics.json` on success |

Keep the entire output root, including failed runs, on persistent storage. A killed
process may leave CLI `run.json` at `initializing`/`running`; the wrapper status and
nonzero/missing return code take precedence. Power loss/SIGKILL can leave the suite
summary `running`; that is incomplete, never passed. No logs or artifacts are
automatically uploaded. Environment inventory excludes credentials and HF tokens.

Remaining validation: run the default eight cells on an actual compatible GPU,
inspect both strict EOS/adherence observations and constrained gates, and retain
environment/log evidence. CPU fixture/mock tests cannot satisfy this gate. Only an
approved, representative labeled fab dataset could support a later accuracy study;
this entrypoint's optional manifest path accepts synthetic stress suites only and
provides no data-upload or fab-data evaluation path.
