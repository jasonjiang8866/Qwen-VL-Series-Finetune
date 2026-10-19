"""Run with python -m defect_benchmark.run --help."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from .backends import BACKENDS
from .core import STRATEGIES, build_prompt, load_config, load_manifest, metrics, parse_output, sha256


def distribution(values):
    values = sorted(x for x in values if x is not None)
    if not values:
        return None
    return {'mean': sum(values) / len(values), 'p50': values[math.ceil(.5 * len(values)) - 1],
            'p95': values[math.ceil(.95 * len(values)) - 1], 'max': max(values)}


def execute(args):
    cfg = load_config(args.config)
    labels = [x['id'] for x in cfg['classes']]
    prompt = build_prompt(cfg, args.prompt_strategy)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)  # never overwrite another run
    rows = load_manifest(args.manifest, labels, out / 'inputs', cfg.get('render', {}))
    synthetic_only = all(x['synthetic'] is True for x in rows)
    real_only = all(x['synthetic'] is False for x in rows)
    evidence = ('cpu_mock' if args.backend == 'mock' else
                'gpu_synthetic' if synthetic_only else
                'gpu_labeled_real_data' if real_only else 'gpu_unspecified_or_mixed_data')
    versions = {}
    for package in ['torch', 'transformers', 'accelerate', 'bitsandbytes', 'Pillow']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    metadata = dict(started_utc=datetime.now(timezone.utc).isoformat(), args=vars(args), config=cfg,
                    config_sha256=sha256(args.config), manifest_sha256=sha256(args.manifest),
                    samples=rows, python=platform.python_version(), versions=versions, code_commit=commit,
                    status='initializing', evidence_level=evidence, synthetic_only=synthetic_only,
                    pipeline_test_only=args.backend == 'mock',
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
    def save_meta():
        (out / 'run.json').write_text(json.dumps(metadata, indent=2))
    save_meta()
    (out / 'prompt.txt').write_text(prompt)
    predictions = []
    try:
        backend = BACKENDS[args.backend](args)
        metadata['backend'] = backend.metadata
        metadata['status'] = 'running'
        save_meta()
        for _ in range(args.warmup):
            backend.predict(rows[0]['image'], prompt, labels)
        with (out / 'predictions.jsonl').open('w') as file:
            for row in rows:
                result = backend.predict(row['image'], prompt, labels)
                prediction = parse_output(result['raw_output'], labels)
                record = dict(**row, **result, prediction=prediction, valid=prediction is not None)
                file.write(json.dumps(record) + '\n')
                file.flush()
                predictions.append(record)
        report = metrics([x['label'] for x in rows], [x['prediction'] for x in predictions], labels)
        report.update(backend=args.backend, model_quality_evaluation=args.backend != 'mock',
                      evidence_level=evidence, synthetic_only=synthetic_only,
                      real_fab_accuracy_validated=False,
                      prompt_strategy=args.prompt_strategy, output_mode=args.output_mode,
                      latency_s=distribution([x['latency_s'] for x in predictions]),
                      generation_s=distribution([x['generation_s'] for x in predictions]),
                      peak_allocated_bytes=distribution([x['peak_allocated_bytes'] for x in predictions]),
                      peak_reserved_bytes=distribution([x['peak_reserved_bytes'] for x in predictions]))
        (out / 'metrics.json').write_text(json.dumps(report, indent=2))
        metadata['status'] = 'complete'
        save_meta()
        print(json.dumps(report, indent=2))
        return report
    except Exception as exc:
        metadata.update(status='failed', error_type=type(exc).__name__, error=str(exc), completed_samples=len(predictions))
        save_meta()
        raise  # runtime/OOM failures must not masquerade as model classification errors


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--backend', choices=BACKENDS, default='transformers')
    p.add_argument('--model', default='Qwen/Qwen2.5-VL-3B-Instruct')
    p.add_argument('--revision', default='main', help='Pin an immutable HF commit for comparisons')
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dtype', choices=['bfloat16', 'float16'], default='bfloat16')
    p.add_argument('--bits', type=int, choices=[4, 8, 16], default=16)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--min-pixels', type=int, default=65536)
    p.add_argument('--max-pixels', type=int, default=262144)
    p.add_argument('--max-new-tokens', type=int, default=32)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--prompt-strategy', choices=STRATEGIES, default='definitions')
    p.add_argument('--output-mode', choices=['constrained', 'strict'], default='constrained')
    return p


def main():
    p = parser()
    args = p.parse_args()
    if not 0 < args.min_pixels <= args.max_pixels or args.warmup < 0 or args.max_new_tokens < 1:
        p.error('invalid pixel/token budget or warmup')
    if not args.device.startswith('cuda:'):
        p.error('device must be explicit cuda:N')
    execute(args)


if __name__ == '__main__':
    main()
