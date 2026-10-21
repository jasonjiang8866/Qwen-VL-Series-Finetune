# Independent review of PR #2

Reviewed on 2026-10-21 (Singapore), against
`7623b489794d3e87265cbdb6324073b25154186e`.
Scope: the unified inference benchmark only. No training changes or merge.

## Confirmed findings and minimal fixes

### P1 — Qwen3 checkpoint defaults can re-enable sampling

Original `backends.py:87-93` creates a `GenerationConfig(do_sample=False, num_beams=1, ...)`
and passes it to `generate` without `use_model_defaults=False`.
Transformers 4.57.1 `_prepare_generation_config` replaces values equal to global defaults
with checkpoint defaults when the checkpoint was saved with Transformers >=4.50.
`False` is the global `do_sample` default, so explicitly placing it inside a
GenerationConfig does **not** protect it. Qwen3-VL-4B-Instruct's generation config has
`do_sample=true`, `temperature=0.7`, `top_p=0.8`, `top_k=20`, version 4.56.0.
This changes both strict and constrained experiments from the documented greedy decoding;
a fixed seed does not restore the intended decoding policy.

Fix: pass `use_model_defaults=False` to `generate`.
A real-library CPU test first reproduces the original override, then exercises the adapter
in both modes and verifies greedy decoding and neutral repetition penalty. The fake-model
contract test also verifies the flag. No real model generation is claimed by these tests.

### P2 — Truncated answers can be scored as complete labels

Original `backends.py:96` only decodes text; `run.py:66` scores any exact label match.
If strict generation exhausts its token budget after a valid short ID that is also a
prefix of a longer answer (for example `a` vs `ab`), the unfinished answer is counted as
valid and can count as correct. The runner cannot distinguish this from EOS completion.

Fix: retain `finish_reason` and `truncated` for actual-backend predictions. Without final
EOS, retain raw text but score the response as invalid, including when its text is a label.
The controlled backend uses only EOS and the token limit as stopping conditions.
Regression tests cover both finish reasons and verify invalid-as-error scoring end to end.
This intentionally makes a label emitted exactly at the limit without EOS invalid; increase
the token budget if that occurs. Mock/custom backends without termination metadata retain
their previous behavior.

### P2 — Recorded config revision did not pin subsequent loads

Original `backends.py:36-50` independently loads config, model and processor with the
requested revision (`main` by default), but records only the config's resolved hash.
If that mutable ref moves during startup, the recorded hash need not identify all loaded
components. This is a code-confirmed race; no actual mismatched checkpoint run was observed.

Fix: reuse the config's resolved Hub commit for subsequent model and processor/tokenizer
loads. Offline/local-files-only and trust-remote-code settings are preserved. Constructor
regression tests verify propagation for both supported dense families and 16/8/4-bit paths.
Local directories can still have a null resolved revision and require external file hashes.
Separate comparison runs still need an explicit immutable `--revision`.

## Processor concern: not a confirmed bug at the pinned version

Transformers 4.57.1 slow and fast Qwen2VL image processors both expose `size` dictionaries
with `shortest_edge`/`longest_edge` pixel-area budgets. They also handle legacy
`min_pixels`/`max_pixels`. Qwen3-VL uses the same fast image processor with checkpoint-specific
patch size 16, not a separate Qwen3 image-processing class. The existing assignments update
both representations without changing patch size. They were left unchanged.

Real-library offline CPU tests execute the backend constructor with real image processors
(slow/fast patch 14 and fast patch 16), while replacing CUDA admission and weight loading.
They preprocess small, large and rectangular RGB images, check pixel budgets with rounding
tolerance, grid divisibility, finite tensors and flattened patch dimensions.

Additionally, an actual `AutoConfig` + `AutoProcessor.from_pretrained` CPU smoke check loaded
the official processor/tokenizer files at these resolved commits, applied the existing
budget settings, rendered the chat template and processed a 448x448 RGB image:

| Checkpoint | Resolved commit | Image processor | Grid | Pixel tensor shape |
| --- | --- | --- | --- | --- |
| Qwen2.5-VL-3B-Instruct | `66285546d2b821cf421d4f5eb2576359d3770cd3` | Qwen2VLImageProcessorFast, patch 14 | `[1,32,32]` | `[1024,1176]` |
| Qwen3-VL-4B-Instruct | `ebb281ec70b05090aa6165b016eac8ec08e71b17` | Qwen2VLImageProcessorFast, patch 16 | `[1,28,28]` | `[784,1536]` |

This checks real processor/tokenizer initialization and preprocessing, **not** model weight
loading, a complete real TransformersBackend initialization, vision-language forward passes,
CUDA kernels, quantized loading or prediction quality. The offline CI suite does not download
these checkpoints' files; it uses real installed image-processor classes and local inputs.

## Other reviewed behavior

- Dense family dispatch is explicit (`qwen2_5_vl`, `qwen3_vl`); unsupported/MoE model types
  fail instead of silently loading an incorrect class. Thinking models remain outside the
  documented comparison; this is not a general VLM adapter.
- Prefix constraints include EOS for completed labels and permit continuing labels with
  overlapping token prefixes. Tokenization collisions, missing EOS and failed round trips
  raise errors. Required output budget includes EOS. Strict mode builds no label trie.
- Official tokenizers at the commits above also passed round-trip/allowed-prefix/EOS checks
  for all five example labels. This is tokenizer validation, not constrained GPU generation.
- Greedy constrained decoding is not full-label likelihood ranking. Naming/token length bias
  remains; low invalid rate is enforced formatting, not semantic accuracy.
- Invalid outputs remain false negatives. Macro-F1 includes absent configured classes as zero;
  this is explicit and tested. No label/ID is passed to the backend's prediction interface.
- CUDA is explicit and single-device, SDPA is requested, bf16 admission is checked, inputs are
  transferred and prompt tokens are trimmed. Generation timing is synchronized. These code
  paths do not establish hardware compatibility or valid latency/VRAM measurements.
- Initialization, warmup and mid-run inference failures preserve `status=failed`; completed
  measured rows survive and no aggregate metrics are produced. Tests now cover all three.
  Input/config validation occurs before the recorded backend run, as documented; invalid
  input may leave an incomplete directory and must not be treated as success.
- Input/prompt/config hashes and evidence levels are retained. Real-data provenance is user
  declared, not independently certified. A clean checkout and immutable model refs are needed
  for reproducibility; `code_commit` alone does not identify uncommitted edits.

## Validation and remaining gates

Local Python 3.12 validation:

- 15 dependency-light CPU tests (including the six prompt/output orchestration cells).
- 2 additional offline tests using actual Transformers 4.57.1, torch 2.8.0+cpu,
  torchvision 0.23.0+cpu and Pillow 11.3.0.
- The official-processor/tokenizer CPU smoke checks described above.

CI retains the Python 3.10/3.12 light suite and six-cell CLI smoke matrix and adds an offline
Python 3.12 real-library CPU job. Check the current PR commit's Actions results separately;
local success is not a claim that remote CI has finished.

Before treating this benchmark as GPU-validated, run **each** dense model at a pinned commit:

1. Load the real checkpoint on the target CUDA GPU; verify bf16/fp16 and 16/8/4-bit modes
   separately (including bitsandbytes support and actual placement/VRAM).
2. Run strict/constrained inference with the real tokenizer, vision inputs and EOS behavior;
   inspect raw output, termination metadata, token grids, and repeatability across processes.
3. Measure synchronized latency/VRAM on the target hardware after warmup. CPU checks supply
   no performance estimates, model fit guarantees or accuracy ranking.
4. Evaluate approved, held-out inspection-coordinate data with domain-agreed classes and
   wafer/lot/time separation. Synthetic fixtures and mock metrics are plumbing evidence only.

## Primary source references

- [Pinned generation config merging implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/generation/utils.py)
- [Pinned slow image processor](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py)
- [Pinned fast image processor](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py)
- [Qwen3 generation config](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/ebb281ec70b05090aa6165b016eac8ec08e71b17/generation_config.json)
- [Qwen3 image processor config](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/ebb281ec70b05090aa6165b016eac8ec08e71b17/preprocessor_config.json)
- [Qwen2.5 image processor config](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/66285546d2b821cf421d4f5eb2576359d3770cd3/preprocessor_config.json)
