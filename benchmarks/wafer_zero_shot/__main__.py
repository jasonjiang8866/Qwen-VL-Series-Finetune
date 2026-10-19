import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

from .backends import MockBackend, TransformersBackend
from .core import messages, metrics, parse_label, render, validate_config
from .synthetic import generate


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def evaluate(records, config, backend, output, warmup=1):
    if not records or len({r['id'] for r in records}) != len(records):
        raise ValueError('Dataset must be nonempty with unique IDs')
    for r in records:
        if r['label'] not in config['labels'] or not isinstance(r.get('synthetic'), bool):
            raise ValueError('Known label and explicit boolean synthetic flag required')
    # Validate every coordinate BEFORE any inference or output creation.
    images = [render(r['coordinates'], radius=r['radius'], center=r['center'], **config['render']) for r in records]
    output.mkdir(parents=True, exist_ok=False)
    (output / 'images').mkdir()
    hashes = []
    for i, image in enumerate(images):
        path = output / 'images' / f'{i:06d}.png'
        image.save(path)
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    dump(output / 'config.json', config)
    summaries = {}
    for strategy in config['strategies']:
        prompt = messages(config, strategy)
        dump(output / f'{strategy}.prompt.json', prompt)
        for _ in range(warmup): backend.generate(images[0], prompt)
        backend.synchronize()
        backend.reset_peak()
        rows = []
        with (output / f'{strategy}.predictions.jsonl').open('w') as stream:
            for i, (record, image) in enumerate(zip(records, images)):
                backend.synchronize()
                start = time.perf_counter()
                raw = backend.generate(image, prompt)
                backend.synchronize()
                elapsed = time.perf_counter() - start
                row = {'id': record['id'], 'truth': record['label'], 'prediction': parse_label(raw, config['labels']),
                       'raw_output': raw, 'latency_seconds': elapsed, 'synthetic': record['synthetic'],
                       'image': f'images/{i:06d}.png', 'image_sha256': hashes[i]}
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                rows.append(row)
        summary = metrics([r['truth'] for r in rows], [r['prediction'] for r in rows], config['labels'])
        times = sorted(r['latency_seconds'] for r in rows)
        summary['latency_seconds'] = {'mean': statistics.mean(times), 'median': statistics.median(times),
                                     'p95_nearest_rank': times[max(0, __import__('math').ceil(.95 * len(times))-1)]}
        summary['peak_gpu_memory'] = backend.peak_memory()
        summary['synthetic_only'] = all(r['synthetic'] for r in records)
        summary['pipeline_test_only'] = isinstance(backend, MockBackend)
        summaries[strategy] = summary
    dump(output / 'metrics.json', summaries)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'configs/default.json')
    sub = parser.add_subparsers(dest='command', required=True)
    synth = sub.add_parser('synthetic')
    synth.add_argument('--output', type=Path, required=True)
    synth.add_argument('--seed', type=int, default=42)
    synth.add_argument('--per-class', type=int, default=4)
    run = sub.add_parser('run')
    run.add_argument('--data', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--backend', choices=['mock', 'transformers'], default='transformers')
    run.add_argument('--model', help='Local checkpoint directory, no downloads')
    run.add_argument('--device', default='cuda:0')
    run.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='bfloat16')
    run.add_argument('--max-new-tokens', type=int, default=32)
    run.add_argument('--warmup', type=int, default=1)
    run.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    config = validate_config(json.loads(args.config.read_text()))
    if args.command == 'synthetic':
        records = generate(config, args.per_class, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            for record in records: stream.write(json.dumps(record) + '\n')
        return
    if args.warmup < 0 or args.max_new_tokens < 1: parser.error('Invalid warmup/token limit')
    if args.output.exists(): parser.error('Output directory already exists; choose a new run directory')
    records = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    if args.backend == 'mock':
        backend = MockBackend(next(iter(config['labels'])))
    else:
        if not args.model or not Path(args.model).is_dir(): parser.error('--model must be an existing local directory')
        from transformers import set_seed
        set_seed(args.seed)
        backend = TransformersBackend(args.model, args.dtype, args.device, args.max_new_tokens)
    evaluate(records, config, backend, args.output, args.warmup)
    versions = {}
    for name in ('Pillow', 'torch', 'transformers', 'accelerate'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    metadata = {'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'python': platform.python_version(), 'packages': versions, 'git_commit': commit.stdout.strip(),
                'dataset_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
                'warning': 'Synthetic data and mock results are pipeline validation only, never real fab performance.',
                'timing': 'Sequential batch size 1; preprocessing + transfer + generate + decode; excludes rendering/model load/warmup.',
                'memory': 'Peak allocated/reserved bytes on selected GPU; includes resident weights; excludes model loading peak.'}
    if args.backend == 'transformers':
        metadata['model_config'] = backend.model.config.to_dict()
        metadata['generation_config'] = backend.model.generation_config.to_dict()
        metadata['gpu'] = backend.torch.cuda.get_device_name(backend.device) if backend.device.type == 'cuda' else None
    dump(args.output / 'run.json', metadata)
    print(f'Results: {args.output}. Synthetic/mock results are NOT fab model performance.')


if __name__ == '__main__': main()
