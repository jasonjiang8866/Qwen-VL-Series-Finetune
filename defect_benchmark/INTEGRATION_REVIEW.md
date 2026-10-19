# Consolidation review: PR #1 and PR #2

Inspected source heads: PR #1 `40daf3c5b8071341894f932f38b2c67fdf2e0e88`;
PR #2 `769849a923f406be194d9313ea11dad3d0a2d304`. Common base:
`512f424e74f94755d774b6e3786457750677048b`.

## Decision

Keep PR #2's `defect_benchmark` as the only framework and CLI. Port PR #1's
fixed-frame renderer, prompt ablation ideas, per-class recall and adapter-contract tests.
Do not merge PR #1 separately. Its branch remains available as design history.

| Area | PR #1 | PR #2 | Unified decision |
| --- | --- | --- | --- |
| Input | Coordinates, explicit radius/center; fixed rendering with geometry rejection | Validated image manifest; procedural renderer only | One manifest supports image OR coordinates; freeze RGB PNGs and hashes per run |
| Prompts | Three strategies, arbitrary format templates | One system prompt, shared ordered classes and decision rules | Three ablations with identical definitions/rules; CLI selects strategy |
| Labels | CENTER/RING/SCRATCH/RANDOM/EMPTY | uniform/center_cluster/edge_ring/scratch/localized_cluster | Keep one configurable PR #2 taxonomy; no silent semantic mapping |
| Output | Exact parsing of free generation | Prefix-constrained or strict free generation | Keep both; strict no longer depends on label tokenization |
| GPU loading | Auto image-text model; claims MoE family support; full model `.to` | Explicit dense Qwen adapters; quantization and image budgets | Keep explicit supported dense families, single GPU and 4/8/16-bit choices |
| Metrics | Accuracy, macro-F1, recall, confusion, invalids | Same essentials, F1/support per class; detailed request metrics | Add precision/recall, retain all classes in macro-F1 denominator |
| Performance | End-to-end latency; per-strategy VRAM peak | Synchronized generation/total time and per-request allocated/reserved memory | Keep PR #2 measurements, warmup and null mock metrics |
| Failure handling | Provenance written after all strategies; failures can lose metadata | Initial/running/failed status, flushed partial predictions | Keep PR #2 lifecycle; invalid data rejected before model load |
| Reproducibility | Seed, image/data hashes, package versions; local-only models | Revision, template/processor metadata, budgets, seed, data hashes | Add exact prompt hash, frozen inputs, explicit synthetic provenance |
| Extensibility | Protocol; eager rendering of entire dataset | Backend registry and per-request measurement dictionary | One backend registry/scoring path; no compatibility wrapper framework |

## Quality and compatibility findings

- Both implementations isolate benchmark dependencies and avoid training changes. Neither
  CPU CI establishes real model-loading or GPU support. PR #1's fake adapter test was a
  useful regression test missing from PR #2; its behavior is now tested on the kept adapter.
- PR #1's AutoModel path advertises Qwen3 MoE without GPU evidence. Do not broaden the
  unified compatibility claim. Dense Qwen2.5-VL and Qwen3-VL are intended, still GPU-unverified.
- PR #2 constructed a label trie even in strict mode. Tokenization collision/round-trip
  checks could prevent the free-generation control. Trie construction is now constrained-only.
- PR #2 used `pad_token_id or eos`, incorrectly replacing a valid pad ID of zero. Fixed and tested.
- Old PR #2 metadata identified mock execution but not synthetic datasets run through a real
  model. Explicit data provenance now separates CPU mock, GPU synthetic, GPU labeled-real
  and unspecified/mixed execution. No run automatically certifies fab validation.
- Different label sets are not interchangeable: PR #1's EMPTY has no PR #2 equivalent;
  PR #2's localized cluster has no PR #1 equivalent; RING and edge_ring differ in scope.
  The example remains illustrative. Empty/mixed maps require an explicitly agreed taxonomy.
- Greedy token-constrained generation is not complete-label likelihood ranking; naming,
  token length and label order must remain fixed across comparisons. Invalid-rate improvement
  in constrained mode is imposed formatting, not better semantic accuracy.
- Coordinate rendering validates geometry, preserves x-right/y-up and physical units, and
  does not autoscale each wafer. Image manifests are canonicalized to RGB PNG; EXIF orientation
  is intentionally ignored. Input styling, processor resizing and dot visibility still matter.
- Pixel budgets, bf16 support, bitsandbytes/CUDA compatibility and model processor behavior
  require target-hardware execution. Model revision `main` is mutable; pin an HF commit or
  retain a local checkpoint checksum inventory. Seeds do not promise cross-GPU bitwise identity.
- Class-absent F1 remains zero and macro-F1 averages all configured classes. No synthetic or
  mock score should be used to rank models. Missing synthetic flags are unknown, never real.

## Validation boundaries

1. CPU: unit tests, synthetic fixtures, six-cell mock experiment and fake model contract.
2. GPU: not performed here. Run pinned dense checkpoints in separate processes, test both
   output modes and quantization settings, inspect transformed dots and collect actual metrics.
3. Fab accuracy: not performed; no enterprise data used. Requires approved taxonomy and
   labeled held-out wafers/lots/time splits, independent domain review and a validation protocol.

No training code, training dependencies, launchers or root configuration are modified.
