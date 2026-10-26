"""Sequential GPU smoke orchestration around the existing benchmark CLI.

--check-only validates the plan and synthetic fixtures without importing torch,
loading checkpoints, or starting subprocess inference. No provider API is used.
"""
import argparse
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .core import STRATEGIES, load_config, load_manifest, metrics, parse_output, sha256
from .run import distribution, parser as benchmark_parser

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'defect_benchmark/configs/example.json'
MANIFEST = ROOT / 'defect_benchmark/fixtures/manifest.jsonl'
MODELS = {
    'qwen25': ('Qwen/Qwen2.5-VL-3B-Instruct', '66285546d2b821cf421d4f5eb2576359d3770cd3'),
    'qwen3': ('Qwen/Qwen3-VL-4B-Instruct', 'ebb281ec70b05090aa6165b016eac8ec08e71b17'),
}


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, help='New directory; never reused or overwritten')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--stress-suite', action='store_true', help='Opt in to synthetic morphology evaluation')
    p.add_argument('--manifest', help='Explicit stress image/coordinate manifest; requires --stress-suite')
    p.add_argument('--stress-replicates', nargs='+', type=int,
                   help='Select complete replicate IDs, retaining all anchors/variants (e.g. 0)')
    p.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    p.add_argument('--bits', nargs='+', type=int, choices=[16, 4], default=[16, 4])
    p.add_argument('--output-modes', nargs='+', choices=['strict', 'constrained'], default=['strict', 'constrained'])
    p.add_argument('--prompt-strategies', nargs='+', choices=STRATEGIES, default=['definitions'])
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--timeout', type=int, default=3600, help='Seconds per cell, including downloads/loading')
    p.add_argument('--max-new-tokens', type=int, default=32)
    p.add_argument('--local-files-only', action='store_true')
    return p


def build_plan(args, out, manifest=None):
    if args.repeats < 1 or args.timeout < 1 or args.max_new_tokens < 1:
        raise ValueError('repeats, timeout and max-new-tokens must be positive')
    if args.stress_suite != bool(args.manifest):
        raise ValueError('--stress-suite and an explicit --manifest must be supplied together')
    if args.stress_replicates is not None and (not args.stress_suite or
                                              any(r < 0 for r in args.stress_replicates)):
        raise ValueError('--stress-replicates requires stress mode and nonnegative replicate IDs')
    manifest = Path(manifest or args.manifest or MANIFEST).expanduser().resolve()
    plan = []
    for key in dict.fromkeys(args.models):
        model, revision = MODELS[key]
        for bits in dict.fromkeys(args.bits):
            for strategy in dict.fromkeys(args.prompt_strategies):
                for mode in dict.fromkeys(args.output_modes):
                    for repeat in range(1, args.repeats + 1):
                        name = f'{key}-bf16-{bits}bit-{strategy}-{mode}-r{repeat:02d}'
                        cli = ['--config', str(CONFIG), '--manifest', str(manifest),
                               '--output', str(out / name / 'benchmark'), '--backend', 'transformers',
                               '--model', model, '--revision', revision, '--device', 'cuda:0',
                               '--dtype', 'bfloat16', '--bits', str(bits), '--seed', '42',
                               '--min-pixels', '65536', '--max-pixels', '262144', '--warmup', '1',
                               '--max-new-tokens', str(args.max_new_tokens),
                               '--prompt-strategy', strategy, '--output-mode', mode]
                        if args.stress_suite:
                            cli.append('--stress-suite')
                        if args.local_files_only:
                            cli.append('--local-files-only')
                        # Catch drift from the existing CLI without importing torch.
                        benchmark_parser().parse_args(cli)
                        plan.append(dict(name=name, model=model, revision=revision, bits=bits,
                                         stress_suite=args.stress_suite, manifest=str(manifest),
                                         mode=mode, strategy=strategy,
                                         command=[sys.executable, '-m', 'defect_benchmark.run', *cli]))
    return plan


def prepare_inputs(args, out):
    """Validate the entire source before selecting whole, pair-closed replicates."""
    cfg = load_config(CONFIG)
    labels = [x['id'] for x in cfg['classes']]
    source = Path(args.manifest or MANIFEST).expanduser().resolve()
    rows = load_manifest(source, labels, out / 'fixture_check', cfg.get('render', {}),
                         stress_suite=args.stress_suite)
    if not all(row['synthetic'] is True for row in rows):
        raise ValueError('Smoke entrypoint accepts only explicit synthetic inputs')
    source_count = len(rows)
    manifest = source
    if args.stress_suite:
        from .stress import validate_suite
        validate_suite(rows, cfg)
        if args.stress_replicates is not None:
            requested = set(args.stress_replicates)
            available = {r['stress']['replicate'] for r in rows}
            if not requested <= available:
                raise ValueError(f'unknown stress replicate IDs: {sorted(requested - available)}')
            rows = [r for r in rows if r['stress']['replicate'] in requested]
        validate_suite(rows, cfg)
        # The child sees exactly these validated RGB pixels, never a mutable source PNG.
        manifest = out / 'selected_manifest.jsonl'
        manifest.write_text(''.join(json.dumps({k: r[k] for k in
                            ('id', 'image', 'label', 'synthetic', 'stress')}) + '\n' for r in rows))
    evidence = dict(config_sha256=sha256(CONFIG), manifest_sha256=sha256(manifest),
                    source_manifest=str(source), source_manifest_sha256=sha256(source),
                    source_samples=source_count, selected_samples=len(rows),
                    labeled_samples=sum(r['label'] is not None for r in rows),
                    exploratory_samples=sum(r['label'] is None for r in rows),
                    stress_suite=args.stress_suite, samples=rows)
    if args.stress_suite:
        evidence.update(replicates=sorted({r['stress']['replicate'] for r in rows}),
                        consistency_pairs=sum(r['stress']['semantic_preserving'] for r in rows))
    write_json(out / 'fixtures.json', evidence)
    return labels, rows, manifest, evidence


def environment():
    versions = {}
    for package in ('torch', 'torchvision', 'transformers', 'accelerate', 'bitsandbytes', 'Pillow', 'huggingface-hub'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    data = dict(python=sys.version, executable=sys.executable, platform=platform.platform(), versions=versions)
    # Do not dump environment variables or pip URLs: they may contain credentials.
    data['installed_packages'] = dict(sorted(
        (dist.metadata['Name'], dist.version) for dist in importlib.metadata.distributions()
        if dist.metadata['Name']))
    data['runtime_settings'] = {key: os.environ.get(key) for key in (
        'CUDA_VISIBLE_DEVICES', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'CUBLAS_WORKSPACE_CONFIG')}
    for key, command in [('code_commit', ['git', 'rev-parse', 'HEAD']),
                         ('code_changes', ['git', 'status', '--porcelain']),
                         ('nvidia_smi', ['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,memory.total,memory.free', '--format=csv'])]:
        try:
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=15)
            data[key] = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        except (OSError, subprocess.TimeoutExpired) as exc:
            data[key] = dict(error=str(exc))
    return data


def preflight():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one CUDA GPU with CUDA_VISIBLE_DEVICES before running')
    torch.cuda.set_device(0)
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('This smoke matrix requires native bf16 support')
    capability = torch.cuda.get_device_capability(0)
    cuda = tuple(int(x) for x in (torch.version.cuda or '0.0').split('.')[:2])
    if capability[0] >= 10 and cuda < (12, 8):
        raise RuntimeError('Blackwell requires a CUDA 12.8+ PyTorch build; see GPU_SMOKE.md')
    # Real kernel admission: version strings alone cannot establish compatibility.
    x = torch.ones((32, 32), dtype=torch.bfloat16, device='cuda:0')
    if not torch.isfinite(x @ x).all().item():
        raise RuntimeError('bf16 CUDA matmul failed')
    torch.cuda.synchronize()
    return dict(gpu=torch.cuda.get_device_name(0), capability=list(capability),
                cuda=torch.version.cuda, torch_arch_list=torch.cuda.get_arch_list(),
                total_memory_bytes=torch.cuda.get_device_properties(0).total_memory)


def inspect_result(cell, directory, labels, samples):
    """Fail closed on missing evidence; no accuracy threshold on synthetic patterns."""
    errors, warnings = [], []
    observed = {}
    def require(condition, message):
        if not condition:
            errors.append(message)
    try:
        meta = json.loads((directory / 'run.json').read_text())
        require(meta.get('status') == 'complete', f"benchmark status={meta.get('status')}: {meta.get('error', '')}")
        backend = meta.get('backend', {})
        require(backend.get('backend') == 'transformers', 'missing real backend evidence')
        require(backend.get('resolved_model_revision') == cell['revision'], 'resolved revision mismatch')
        require(meta.get('evidence_level') == 'gpu_synthetic' and meta.get('synthetic_only') is True,
                'expected synthetic-only GPU evidence')
        actual = meta.get('args', {})
        expected = dict(model=cell['model'], revision=cell['revision'], bits=cell['bits'],
                        dtype='bfloat16', device='cuda:0', output_mode=cell['mode'], prompt_strategy=cell['strategy'])
        require(all(actual.get(k) == v for k, v in expected.items()), 'CLI settings mismatch')
        require(backend.get('parameter_devices') == ['cuda:0'], 'model parameters not exclusively on cuda:0')
        require('torch.bfloat16' in backend.get('parameter_dtypes', []), 'missing bf16 parameters')
        if cell['bits'] == 4:
            require(backend.get('loaded_in_4bit') is True and backend.get('linear4bit_modules', 0) > 0,
                    '4-bit weights were not observed')
            require(backend.get('quantized_compute_dtypes') == ['torch.bfloat16'], 'NF4 compute is not bf16')
        else:
            require(backend.get('loaded_in_4bit') is False and backend.get('linear4bit_modules') == 0,
                    'unexpected quantization in bf16 cell')
        records = [json.loads(line) for line in (directory / 'predictions.jsonl').read_text().splitlines()]
        require([r.get('id') for r in records] == [r['id'] for r in samples], 'missing, duplicate or reordered samples')
        eos_count, length_count, invalid_count = 0, 0, 0
        for index, row in enumerate(records):
            prefix = f'sample {index}: '
            ids, eos = row.get('generated_token_ids'), row.get('eos_token_id')
            tokens_ok = (isinstance(ids, list) and bool(ids) and all(type(t) is int for t in ids)
                         and type(eos) is int)
            require(tokens_ok, prefix + 'missing token-level EOS evidence')
            if not tokens_ok:
                continue
            ended = ids[-1] == eos
            eos_count += int(ended)
            length_count += int(not ended)
            require(eos not in ids[:-1], prefix + 'generation continued after EOS')
            require(row.get('output_tokens') == len(ids), prefix + 'token count mismatch')
            require(row.get('finish_reason') == ('eos' if ended else 'length') and row.get('truncated') is (not ended),
                    prefix + 'EOS/truncation flags inconsistent')
            require(0 < len(ids) <= actual.get('max_new_tokens', 0), prefix + 'token budget violated')
            if not ended:
                require(len(ids) == actual.get('max_new_tokens'), prefix + 'non-EOS stop before token limit')
            prediction = parse_output(row.get('raw_output', ''), labels) if ended else None
            require(row.get('prediction') == prediction and row.get('valid') is (prediction is not None),
                    prefix + 'parsing/truncation scoring mismatch')
            invalid_count += int(prediction is None)
            if cell['mode'] == 'constrained':
                require(ended and prediction is not None, prefix + 'constrained output must be label + EOS')
            for metric in ('latency_s', 'generation_s', 'peak_allocated_bytes', 'peak_reserved_bytes'):
                value = row.get(metric)
                require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                        prefix + f'missing/invalid {metric}')
        if length_count or invalid_count:
            warnings.append(f'{length_count} length stops, {invalid_count} invalid labels; inspect raw outputs')
        # Strict adherence is a separate observation, not model-loading success.
        report = json.loads((directory / 'metrics.json').read_text())
        require(report.get('real_fab_accuracy_validated') is False, 'invalid fab accuracy claim')
        observed = dict(eos_count=eos_count, length_count=length_count, invalid_count=invalid_count,
                        all_outputs_eos=bool(records) and eos_count == len(records),
                        strict_adherence_pass=(invalid_count == 0) if cell['mode'] == 'strict' else None,
                        measurements={k: report.get(k) for k in (
                            'latency_s', 'generation_s', 'peak_allocated_bytes', 'peak_reserved_bytes')},
                        model_load_and_processor_s=backend.get('model_load_and_processor_s'))
        if cell.get('stress_suite'):
            from .stress import summarize
            require(actual.get('stress_suite') is True and actual.get('manifest') == cell['manifest'],
                    'stress manifest/mode mismatch')
            require(meta.get('manifest_sha256') == cell['manifest_sha256'] and
                    meta.get('config_sha256') == cell['config_sha256'], 'input/config hash mismatch')
            for row, sample in zip(records, samples):
                require(all(row.get(k) == sample[k] for k in
                            ('label', 'synthetic', 'stress', 'pixels_sha256')),
                        f"stress input provenance mismatch: {sample['id']}")
            # Recompute from raw per-sample evidence; missing/incorrect aggregate files fail closed.
            stress = json.loads((directory / 'stress_metrics.json').read_text())
            recomputed = summarize(records, labels)
            require(all(stress.get(k) == v for k, v in recomputed.items()), 'stress metrics mismatch')
            require(stress.get('evidence_level') == 'gpu_synthetic' and
                    stress.get('pipeline_test_only') is False, 'missing stress GPU evidence')
            labeled = [r for r in records if r['label'] is not None]
            exploratory = [r for r in records if r['label'] is None]
            classification = metrics([r['label'] for r in labeled], [r['prediction'] for r in labeled], labels)
            require(all(report.get(k) == v for k, v in classification.items()), 'classification metrics mismatch')
            require(report.get('total_samples') == len(samples) and
                    report.get('labeled_samples') == len(labeled) and
                    report.get('exploratory_samples') == len(exploratory), 'scoring denominators mismatch')
            saved_exploratory = [json.loads(line) for line in
                                (directory / 'exploratory_predictions.jsonl').read_text().splitlines()]
            require(saved_exploratory == exploratory, 'exploratory predictions missing or mislabeled')
            measurements = {k: distribution([r[k] for r in records]) for k in
                            ('latency_s', 'generation_s', 'peak_allocated_bytes', 'peak_reserved_bytes')}
            require(all(report.get(k) == v for k, v in measurements.items()), 'measurement summaries mismatch')
            def invalid_rate(rows):
                return sum(r['prediction'] not in labels for r in rows) / len(rows) if rows else None
            observed.update(classification=classification,
                            morphology={k: v for k, v in stress.items() if k != 'pairs'},
                            invalid_output_rates=dict(all_samples=invalid_rate(records),
                                                      labeled=invalid_rate(labeled),
                                                      exploratory=invalid_rate(exploratory)))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
    return dict(checks_passed=not errors, errors=errors, warnings=warnings, observed=observed)


def run_cell(cell, out, timeout):
    directory = out / cell['name']
    directory.mkdir()
    write_json(directory / 'command.json', cell['command'])
    start = time.monotonic()
    result = dict(name=cell['name'], status='running', returncode=None)
    write_json(directory / 'status.json', result)
    try:
        with (directory / 'stdout.log').open('w') as stdout, (directory / 'stderr.log').open('w') as stderr:
            completed = subprocess.run(cell['command'], cwd=ROOT, stdout=stdout, stderr=stderr, timeout=timeout)
        result.update(returncode=completed.returncode, status='executed' if completed.returncode == 0 else 'failed')
        if completed.returncode:
            result['error'] = f'CLI exited {completed.returncode}; see stderr.log and benchmark/run.json'
    except subprocess.TimeoutExpired:
        result.update(status='failed', error=f'timeout after {timeout}s; child killed and reaped')
    except OSError as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    result['wall_time_s'] = time.monotonic() - start
    write_json(directory / 'status.json', result)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    out = Path(args.output).expanduser().resolve()
    plan = build_plan(args, out)
    out.mkdir(parents=True, exist_ok=False)
    summary = dict(status='initializing', evidence_level='not_executed',
                   real_fab_accuracy_validated=False, cells=[], planned_cells=len(plan),
                   started_utc=datetime.now(timezone.utc).isoformat())
    write_json(out / 'plan.json', plan)
    write_json(out / 'summary.json', summary)
    try:
        write_json(out / 'environment.json', environment())
        labels, samples, manifest, evidence = prepare_inputs(args, out)
        plan = build_plan(args, out, manifest)
        directories = [benchmark_parser().parse_args(c['command'][3:]).output for c in plan]
        if len(set(directories)) != len(plan):
            raise ValueError('experiment directories must be unique')
        for cell in plan:
            cell.update(config_sha256=evidence['config_sha256'], manifest_sha256=evidence['manifest_sha256'])
        write_json(out / 'plan.json', plan)
        summary.update(stress_suite=args.stress_suite, selected_samples=len(samples),
                       labeled_samples=evidence['labeled_samples'],
                       exploratory_samples=evidence['exploratory_samples'],
                       planned_predictions=len(plan) * len(samples),
                       excluded_warmup_predictions=len(plan))
        if args.stress_suite:
            summary.update(replicates=evidence['replicates'], consistency_pairs=evidence['consistency_pairs'])
        if args.check_only:
            summary['status'] = 'configuration_checked'
            print(f'Configuration checked: {len(plan)} cells; no checkpoint/CUDA inference performed')
            return 0
        write_json(out / 'preflight.json', preflight())
        summary.update(status='running', evidence_level='gpu_synthetic_smoke')
        for cell in plan:
            print(f"Running {cell['name']}", flush=True)
            result = run_cell(cell, out, args.timeout)
            result.update(inspect_result(cell, out / cell['name'] / 'benchmark', labels, samples))
            result['status'] = 'passed' if result['returncode'] == 0 and result['checks_passed'] else 'failed'
            write_json(out / cell['name'] / 'status.json', result)
            summary['cells'].append(result)
            write_json(out / 'summary.json', summary)
        summary['status'] = 'passed' if all(c['status'] == 'passed' for c in summary['cells']) else 'failed'
        return 0 if summary['status'] == 'passed' else 1
    except KeyboardInterrupt:
        summary.update(status='interrupted', error='Interrupted by user; partial artifacts retained')
        return 130
    except Exception as exc:
        summary.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        print(summary['error'], file=sys.stderr)
        return 1
    finally:
        summary['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(out / 'summary.json', summary)
        print(f"{summary['status']}: {out / 'summary.json'}")


if __name__ == '__main__':
    sys.exit(main())
