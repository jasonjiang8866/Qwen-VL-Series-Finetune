"""Offline orchestration/result-validation tests; all GPU evidence here is fabricated."""
import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from defect_benchmark import gpu_smoke as smoke


class SmokeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = smoke.parser().parse_args(['--output', str(self.root / 'suite')])
        self.plan = smoke.build_plan(self.args, self.root / 'suite')

    def test_matrix_pins_and_isolation(self):
        self.assertEqual(len(self.plan), 8)
        self.assertEqual(len({c['name'] for c in self.plan}), 8)
        for cell in self.plan:
            args = smoke.benchmark_parser().parse_args(cell['command'][3:])
            self.assertEqual(args.revision, smoke.MODELS[cell['name'].split('-')[0]][1])
            self.assertEqual(len(args.revision), 40)
            self.assertEqual(args.backend, 'transformers')
            self.assertEqual(args.dtype, 'bfloat16')
            self.assertEqual(args.device, 'cuda:0')
            self.assertEqual(args.warmup, 1)
            self.assertIn(cell['name'], args.output)
        self.args.prompt_strategies = list(smoke.STRATEGIES)
        self.args.repeats = 2
        self.args.local_files_only = True
        expanded = smoke.build_plan(self.args, self.root)
        self.assertEqual(len(expanded), 48)
        self.assertTrue(all('--local-files-only' in c['command'] for c in expanded))
        self.args.timeout = 0
        with self.assertRaises(ValueError):
            smoke.build_plan(self.args, self.root)

    def test_check_only_never_loads_gpu_and_refuses_overwrite(self):
        with patch.object(smoke, 'preflight', side_effect=AssertionError('GPU forbidden')), \
             patch.object(smoke, 'run_cell', side_effect=AssertionError('inference forbidden')), \
             patch.object(smoke, 'environment', return_value={}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(smoke.main(['--output', self.args.output, '--check-only']), 0)
            with self.assertRaises(FileExistsError):
                smoke.main(['--output', self.args.output, '--check-only'])
        summary = json.loads((Path(self.args.output) / 'summary.json').read_text())
        self.assertEqual(summary['status'], 'configuration_checked')
        self.assertEqual(summary['evidence_level'], 'not_executed')
        self.assertEqual(summary['cells'], [])

    def result_fixture(self, cell):
        directory = self.root / 'artifacts'
        directory.mkdir(exist_ok=True)
        meta = dict(status='complete', evidence_level='gpu_synthetic', synthetic_only=True,
                    backend=dict(backend='transformers', resolved_model_revision=cell['revision'],
                                 parameter_devices=['cuda:0'], parameter_dtypes=['torch.bfloat16'],
                                 loaded_in_4bit=cell['bits'] == 4, linear4bit_modules=10 if cell['bits'] == 4 else 0,
                                 quantized_compute_dtypes=['torch.bfloat16'] if cell['bits'] == 4 else []),
                    args=dict(model=cell['model'], revision=cell['revision'], dtype='bfloat16',
                              bits=cell['bits'], device='cuda:0', output_mode=cell['mode'],
                              prompt_strategy=cell['strategy'], max_new_tokens=2))
        record = dict(id='sample', raw_output='uniform', prediction='uniform', valid=True,
                      generated_token_ids=[10, 99], eos_token_id=99, output_tokens=2,
                      finish_reason='eos', truncated=False, latency_s=.2, generation_s=.1,
                      peak_allocated_bytes=1024, peak_reserved_bytes=2048)
        smoke.write_json(directory / 'run.json', meta)
        smoke.write_json(directory / 'metrics.json', dict(real_fab_accuracy_validated=False))
        (directory / 'predictions.jsonl').write_text(json.dumps(record) + '\n')
        return directory, meta, record

    def inspect(self, cell, directory):
        return smoke.inspect_result(cell, directory, ['uniform'], [{'id': 'sample'}])

    def test_valid_matrix_artifact_contract(self):
        for cell in self.plan:
            directory, _, _ = self.result_fixture(cell)
            result = self.inspect(cell, directory)
            self.assertTrue(result['checks_passed'], result)
            self.assertTrue(result['observed']['all_outputs_eos'])

    def test_strict_truncation_never_scored_as_label(self):
        cell = next(c for c in self.plan if c['mode'] == 'strict')
        directory, _, row = self.result_fixture(cell)
        row.update(generated_token_ids=[10, 11], finish_reason='length', truncated=True)
        (directory / 'predictions.jsonl').write_text(json.dumps(row) + '\n')
        self.assertFalse(self.inspect(cell, directory)['checks_passed'])
        row.update(prediction=None, valid=False)
        (directory / 'predictions.jsonl').write_text(json.dumps(row) + '\n')
        result = self.inspect(cell, directory)
        self.assertTrue(result['checks_passed'])
        self.assertEqual(result['observed']['length_count'], 1)
        self.assertFalse(result['observed']['strict_adherence_pass'])
        self.assertFalse(result['observed']['all_outputs_eos'])
        constrained = dict(cell, mode='constrained')
        self.assertFalse(self.inspect(constrained, directory)['checks_passed'])

    def test_missing_or_false_gpu_evidence_fails(self):
        cell = next(c for c in self.plan if c['bits'] == 4)
        for field, value in [('resolved_model_revision', 'main'), ('loaded_in_4bit', False),
                             ('linear4bit_modules', 0), ('parameter_devices', ['cpu']),
                             ('quantized_compute_dtypes', ['torch.float16']), ('parameter_dtypes', [])]:
            with self.subTest(field=field):
                directory, meta, _ = self.result_fixture(cell)
                meta['backend'][field] = value
                smoke.write_json(directory / 'run.json', meta)
                self.assertFalse(self.inspect(cell, directory)['checks_passed'])
        for field, value in [('latency_s', None), ('generation_s', float('nan')),
                             ('peak_allocated_bytes', 0), ('generated_token_ids', []),
                             ('generated_token_ids', [99, 10]), ('truncated', True), ('output_tokens', 3)]:
            with self.subTest(field=field):
                directory, _, row = self.result_fixture(cell)
                row[field] = value
                (directory / 'predictions.jsonl').write_text(json.dumps(row) + '\n')
                self.assertFalse(self.inspect(cell, directory)['checks_passed'])
        (directory / 'predictions.jsonl').unlink()
        self.assertFalse(self.inspect(cell, directory)['checks_passed'])

    def test_timeout_and_nonzero_keep_logs(self):
        for i, effect in enumerate([subprocess.TimeoutExpired(['test'], 1),
                                    subprocess.CompletedProcess(['test'], 7)]):
            out = self.root / str(i)
            out.mkdir()
            kwargs = {'side_effect': effect} if isinstance(effect, Exception) else {'return_value': effect}
            with patch.object(smoke.subprocess, 'run', **kwargs):
                result = smoke.run_cell(self.plan[0], out, 1)
            self.assertEqual(result['status'], 'failed')
            for name in ('command.json', 'stdout.log', 'stderr.log', 'status.json'):
                self.assertTrue((out / self.plan[0]['name'] / name).exists())

    def test_failure_continues_matrix_and_returns_nonzero(self):
        calls = []
        def fake_run(cell, out, timeout):
            calls.append(cell['name'])
            (out / cell['name']).mkdir()
            return dict(name=cell['name'], returncode=7 if len(calls) == 1 else 0)
        with patch.object(smoke, 'environment', return_value={}), patch.object(smoke, 'preflight', return_value={}), \
             patch.object(smoke, 'run_cell', side_effect=fake_run), \
             patch.object(smoke, 'inspect_result', return_value=dict(checks_passed=True)), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(smoke.main(['--output', self.args.output]), 1)
        summary = json.loads((Path(self.args.output) / 'summary.json').read_text())
        self.assertEqual(len(calls), 8)
        self.assertEqual(summary['status'], 'failed')
        self.assertEqual(summary['cells'][0]['status'], 'failed')
        self.assertEqual(summary['cells'][-1]['status'], 'passed')

    def test_no_cuda_preserves_failed_preflight(self):
        with patch.object(smoke, 'environment', return_value={}), \
             patch.object(smoke, 'preflight', side_effect=RuntimeError('no CUDA')), \
             patch.object(smoke, 'run_cell', side_effect=AssertionError('must not run')), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(smoke.main(['--output', self.args.output]), 1)
        summary = json.loads((Path(self.args.output) / 'summary.json').read_text())
        self.assertEqual(summary['evidence_level'], 'not_executed')
        self.assertIn('no CUDA', summary['error'])


if __name__ == '__main__':
    unittest.main()
