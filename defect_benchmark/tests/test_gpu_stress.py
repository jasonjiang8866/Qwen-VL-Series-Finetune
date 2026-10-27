"""CPU-only stress orchestration contracts; fabricated GPU artifacts are NOT inference evidence."""
import contextlib
import io
import json
import re
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from defect_benchmark import gpu_smoke as smoke
from defect_benchmark.core import load_config, load_manifest, metrics
from defect_benchmark.run import distribution, execute
from defect_benchmark.stress import generate, summarize, validate_suite


class GPUStressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory()
        cls.dataset = Path(cls.shared.name) / 'dataset'
        generate(cls.dataset)

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / 'out'

    def cli(self, *extra, manifest='manifest'):
        return ['--output', str(self.out), '--stress-suite', '--manifest',
                str(self.dataset / f'{manifest}.jsonl'), *extra]

    def check(self, cli, expected=0):
        with patch.object(smoke, 'preflight', side_effect=AssertionError('CUDA forbidden')), \
             patch.object(smoke, 'run_cell', side_effect=AssertionError('inference forbidden')), \
             patch.object(smoke, 'environment', return_value={}), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(smoke.main([*cli, '--check-only']), expected)
        return json.loads((self.out / 'summary.json').read_text())

    def test_full_plan_both_manifests_and_prompt_matrix(self):
        pixels = []
        for manifest in ('manifest', 'coordinates'):
            self.out = self.root / manifest
            summary = self.check(self.cli('--prompt-strategies', *smoke.STRATEGIES, manifest=manifest))
            self.assertEqual(summary['status'], 'configuration_checked')
            self.assertEqual(summary['evidence_level'], 'not_executed')
            self.assertIs(summary['real_fab_accuracy_validated'], False)
            self.assertEqual(summary['planned_cells'], 24)
            self.assertEqual(summary['selected_samples'], 204)
            self.assertEqual(summary['labeled_samples'], 113)
            self.assertEqual(summary['exploratory_samples'], 91)
            self.assertEqual(summary['consistency_pairs'], 98)
            self.assertEqual(summary['planned_predictions'], 204 * 24)
            self.assertEqual(summary['cells'], [])
            self.assertFalse((self.out / 'preflight.json').exists())
            plan = json.loads((self.out / 'plan.json').read_text())
            dirs = []
            for cell in plan:
                args = smoke.benchmark_parser().parse_args(cell['command'][3:])
                self.assertTrue(args.stress_suite)
                self.assertEqual(args.manifest, str(self.out / 'selected_manifest.jsonl'))
                self.assertEqual(len(args.revision), 40)
                self.assertEqual(args.dtype, 'bfloat16')
                self.assertEqual(args.device, 'cuda:0')
                self.assertEqual(cell['manifest_sha256'], smoke.sha256(args.manifest))
                dirs.append(args.output)
            self.assertEqual(len(set(dirs)), 24)
            self.assertEqual({c['bits'] for c in plan}, {4, 16})
            self.assertEqual({c['mode'] for c in plan}, {'strict', 'constrained'})
            self.assertEqual({c['model'] for c in plan}, {m[0] for m in smoke.MODELS.values()})
            evidence = json.loads((self.out / 'fixtures.json').read_text())
            pixels.append([r['pixels_sha256'] for r in evidence['samples']])
        self.assertEqual(*pixels)

    def test_small_selection_keeps_complete_replicate_and_references(self):
        summary = self.check(self.cli('--stress-replicates', '0', '--models', 'qwen3',
                                      '--bits', '4', '--output-modes', 'strict', 'constrained'))
        self.assertEqual(summary['planned_cells'], 2)
        self.assertEqual(summary['selected_samples'], 68)
        self.assertEqual(summary['planned_predictions'], 136)
        self.assertEqual(summary['replicates'], [0])
        rows = json.loads((self.out / 'fixtures.json').read_text())['samples']
        validate_suite(rows, load_config(smoke.CONFIG))
        ids = {r['id'] for r in rows}
        self.assertEqual(sum(r['stress']['role'] == 'anchor' for r in rows), 5)
        self.assertTrue(all(r['stress']['reference_id'] in ids for r in rows
                            if r['stress']['role'] == 'variant'))
        selected = load_manifest(self.out / 'selected_manifest.jsonl',
                                 [c['id'] for c in load_config(smoke.CONFIG)['classes']],
                                 self.root / 'reloaded', load_config(smoke.CONFIG)['render'], stress_suite=True)
        self.assertEqual([r['pixels_sha256'] for r in rows], [r['pixels_sha256'] for r in selected])

    def test_opt_in_required_and_no_silent_manifest_override(self):
        cases = [ ['--stress-suite'], ['--manifest', str(self.dataset / 'manifest.jsonl')],
                  ['--stress-replicates', '0'],
                  ['--stress-suite', '--manifest', str(self.dataset / 'manifest.jsonl'),
                   '--stress-replicates', '-1'] ]
        for cli in cases:
            with self.subTest(cli=cli), self.assertRaises(ValueError):
                args = smoke.parser().parse_args(['--output', str(self.out), *cli])
                smoke.build_plan(args, self.out)

    def test_default_stays_five_images(self):
        summary = self.check(['--output', str(self.out)])
        self.assertEqual(summary['planned_cells'], 8)
        self.assertEqual(summary['selected_samples'], 5)
        self.assertFalse(summary['stress_suite'])
        for cell in json.loads((self.out / 'plan.json').read_text()):
            self.assertNotIn('--stress-suite', cell['command'])
            self.assertEqual(cell['manifest'], str(smoke.MANIFEST))

    def test_documented_cpu_plans(self):
        """Run the actual copyable plan commands; forbid CUDA/inference via check()."""
        guides = {
            'GPU_SMOKE.md': [(8, 5, 40, False)],
            'STRESS_TEST.md': [(24, 204, 4896, True), (2, 68, 136, True)],
        }
        for guide, expected in guides.items():
            text = (smoke.ROOT / 'defect_benchmark' / guide).read_text()
            commands = [shlex.split(line) for block in re.findall(r'```bash\n(.*?)```', text, re.S)
                        for line in block.replace('\\\n', ' ').splitlines()
                        if line.startswith('python -m defect_benchmark.gpu_smoke --check-only')]
            self.assertEqual(len(commands), len(expected), guide)
            for index, (command, (cells, samples, predictions, stress)) in enumerate(zip(commands, expected)):
                with self.subTest(guide=guide, command=command):
                    argv = command[3:]
                    self.assertIn('--check-only', argv)
                    if '--manifest' in argv:
                        pos = argv.index('--manifest') + 1
                        argv[pos] = argv[pos].replace('/tmp/ddm-stress-v1', str(self.dataset))
                    self.out = self.root / f'{guide}-{index}'
                    argv[argv.index('--output') + 1] = str(self.out)
                    summary = self.check(argv)
                    self.assertEqual(summary['status'], 'configuration_checked')
                    self.assertEqual(summary['evidence_level'], 'not_executed')
                    self.assertEqual(summary['planned_cells'], cells)
                    self.assertEqual(summary['selected_samples'], samples)
                    self.assertEqual(summary['planned_predictions'], predictions)
                    self.assertEqual(summary['stress_suite'], stress)
                    self.assertEqual(summary['cells'], [])
                    self.assertFalse((self.out / 'preflight.json').exists())

    def write_corrupt_manifest(self, mutate):
        rows = [json.loads(s) for s in (self.dataset / 'manifest.jsonl').read_text().splitlines()]
        # Keep source paths valid when moving the manifest out of its dataset directory.
        for r in rows:
            r['image'] = str(self.dataset / r['image'])
        mutate(rows)
        source = self.root / 'corrupt.jsonl'
        source.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        return source

    def test_invalid_inputs_fail_before_preflight_even_when_not_selected(self):
        mutations = {
            'pixels': lambda rows: rows[-1]['stress'].update(pixels_sha256='0' * 64),
            'missing_pixels': lambda rows: rows[-1]['stress'].pop('pixels_sha256'),
            'config': lambda rows: rows[1]['stress'].update(config_sha256='0' * 64),
            'pair': lambda rows: rows[1]['stress'].update(reference_id='missing'),
            'cross_replicate_pair': lambda rows: rows[1]['stress'].update(reference_id=rows[68]['id']),
            'unknown_class': lambda rows: rows[0].update(label='unknown'),
            'exploratory_label': lambda rows: next(r for r in rows if r['label'] is None).update(label='uniform'),
            'real_data': lambda rows: rows[0].update(synthetic=False),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                self.out = self.root / name
                source = self.write_corrupt_manifest(mutation)
                summary = self.check(['--output', str(self.out), '--stress-suite', '--manifest', str(source),
                                      '--stress-replicates', '0'], expected=1)
                self.assertEqual(summary['status'], 'failed')
                self.assertEqual(summary['evidence_level'], 'not_executed')
                self.assertEqual(summary['cells'], [])

    def test_unknown_replicate_fails(self):
        summary = self.check(self.cli('--stress-replicates', '99'), expected=1)
        self.assertIn('unknown stress replicate', summary['error'])

    def test_frozen_manifest_does_not_reuse_source_images(self):
        self.check(self.cli('--stress-replicates', '0'))
        rows = [json.loads(s) for s in (self.out / 'selected_manifest.jsonl').read_text().splitlines()]
        for row in rows:
            self.assertTrue(Path(row['image']).is_relative_to(self.out / 'fixture_check'))
            self.assertNotIn('coordinates', row)

    def test_selected_plan_executes_existing_cli_with_cpu_mock_only(self):
        self.check(self.cli('--stress-replicates', '0', '--models', 'qwen3', '--bits', '4',
                            '--output-modes', 'strict', 'constrained'))
        for cell in json.loads((self.out / 'plan.json').read_text()):
            args = smoke.benchmark_parser().parse_args(cell['command'][3:])
            args.backend = 'mock'  # CPU plumbing check; never pretend this is GPU validation.
            with contextlib.redirect_stdout(io.StringIO()):
                report = execute(args)
            self.assertEqual(report['evidence_level'], 'cpu_mock')
            self.assertFalse(report['real_fab_accuracy_validated'])
            self.assertEqual(report['total_samples'], 68)
            self.assertEqual(report['n'], report['labeled_samples'])
            self.assertEqual(report['labeled_samples'] + report['exploratory_samples'], 68)
            stress = json.loads((Path(args.output) / 'stress_metrics.json').read_text())
            self.assertEqual(stress['evidence_level'], 'cpu_mock')
            self.assertFalse(stress['model_quality_evaluation'])
            # CPU mock metrics cannot pass the real-GPU gate.
            samples = json.loads((self.out / 'fixtures.json').read_text())['samples']
            labels = [c['id'] for c in load_config(smoke.CONFIG)['classes']]
            self.assertFalse(smoke.inspect_result(cell, Path(args.output), labels, samples)['checks_passed'])

    def fabricated_artifacts(self):
        """Only test the evidence validator; these objects never came from a GPU."""
        self.check(self.cli('--stress-replicates', '0', '--models', 'qwen25', '--bits', '4',
                            '--output-modes', 'strict'))
        cell = json.loads((self.out / 'plan.json').read_text())[0]
        evidence = json.loads((self.out / 'fixtures.json').read_text())
        samples = evidence['samples']
        labels = [c['id'] for c in load_config(smoke.CONFIG)['classes']]
        records = [dict(r, raw_output='uniform', prediction='uniform', valid=True,
                        generated_token_ids=[10, 99], eos_token_id=99, output_tokens=2,
                        finish_reason='eos', truncated=False, latency_s=.2, generation_s=.1,
                        peak_allocated_bytes=1024, peak_reserved_bytes=2048) for r in samples]
        # One exploratory invalid output: must affect overall invalid rate, never accuracy.
        next(r for r in records if r['label'] is None).update(raw_output='explanation',
                                                            prediction=None, valid=False)
        args = vars(smoke.benchmark_parser().parse_args(cell['command'][3:]))
        meta = dict(status='complete', evidence_level='gpu_synthetic', synthetic_only=True,
                    args=args, config_sha256=cell['config_sha256'], manifest_sha256=cell['manifest_sha256'],
                    backend=dict(backend='transformers', resolved_model_revision=cell['revision'],
                                 parameter_devices=['cuda:0'], parameter_dtypes=['torch.bfloat16'],
                                 loaded_in_4bit=True, linear4bit_modules=10,
                                 quantized_compute_dtypes=['torch.bfloat16']))
        labeled = [r for r in records if r['label'] is not None]
        report = metrics([r['label'] for r in labeled], [r['prediction'] for r in labeled], labels)
        report.update(real_fab_accuracy_validated=False, total_samples=len(samples), labeled_samples=len(labeled),
                      exploratory_samples=len(samples) - len(labeled))
        report.update({k: distribution([r[k] for r in records]) for k in
                       ('latency_s', 'generation_s', 'peak_allocated_bytes', 'peak_reserved_bytes')})
        stress = summarize(records, labels)
        stress.update(evidence_level='gpu_synthetic', pipeline_test_only=False)
        directory = self.out / cell['name'] / 'benchmark'
        directory.mkdir(parents=True)
        for file, value in [('run.json', meta), ('metrics.json', report), ('stress_metrics.json', stress)]:
            smoke.write_json(directory / file, value)
        for file, value in [('predictions.jsonl', records), ('exploratory_predictions.jsonl',
                                                           [r for r in records if r['label'] is None])]:
            (directory / file).write_text(''.join(json.dumps(r) + '\n' for r in value))
        return cell, directory, labels, samples

    def test_result_summary_separates_accuracy_consistency_invalid_and_cost(self):
        cell, directory, labels, samples = self.fabricated_artifacts()
        result = smoke.inspect_result(cell, directory, labels, samples)
        self.assertTrue(result['checks_passed'], result)
        observed = result['observed']
        self.assertEqual(observed['classification']['n'], sum(r['label'] is not None for r in samples))
        self.assertEqual(observed['morphology']['overall']['consistency']['consistency'], 1)
        self.assertLess(observed['classification']['accuracy'], 1)
        self.assertEqual(observed['invalid_output_rates']['labeled'], 0)
        self.assertGreater(observed['invalid_output_rates']['exploratory'], 0)
        self.assertAlmostEqual(observed['measurements']['latency_s']['mean'], .2)

    def test_missing_or_modified_stress_evidence_fails_closed(self):
        cell, directory, labels, samples = self.fabricated_artifacts()
        for file, mutate in [
            ('run.json', lambda x: x.update(manifest_sha256='wrong')),
            ('run.json', lambda x: x['args'].update(stress_suite=False)),
            ('metrics.json', lambda x: x.update(n=68)),
            ('metrics.json', lambda x: x.update(latency_s=None)),
            ('stress_metrics.json', lambda x: x['overall']['consistency'].update(pairs=999)),
        ]:
            with self.subTest(file=file):
                original = (directory / file).read_text()
                value = json.loads(original)
                mutate(value)
                smoke.write_json(directory / file, value)
                self.assertFalse(smoke.inspect_result(cell, directory, labels, samples)['checks_passed'])
                (directory / file).write_text(original)
        (directory / 'exploratory_predictions.jsonl').unlink()
        self.assertFalse(smoke.inspect_result(cell, directory, labels, samples)['checks_passed'])


if __name__ == '__main__':
    unittest.main()
