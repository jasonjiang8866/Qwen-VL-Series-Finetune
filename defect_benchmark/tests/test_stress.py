"""CPU-only metamorphic and accounting tests; no model weights or CUDA."""
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from defect_benchmark.core import load_config, load_manifest
from defect_benchmark.run import execute, parser
from defect_benchmark.stress import CONFIG, generate, summarize, validate_suite
from defect_benchmark.synthetic import generate as generate_smoke


class StressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.manifest = generate(cls.root / 'suite', seed=42, replicates=1)
        cls.rows = [json.loads(s) for s in (cls.manifest.parent / 'coordinates.jsonl').read_text().splitlines()]
        cls.cfg = load_config(CONFIG)
        cls.labels = [c['id'] for c in cls.cfg['classes']]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_reproducible_artifacts_and_independent_replicates(self):
        other = generate(self.root / 'same', seed=42, replicates=1)
        for name in ('manifest.jsonl', 'coordinates.jsonl', 'provenance.json'):
            self.assertEqual((other.parent / name).read_bytes(), (self.manifest.parent / name).read_bytes())
        for image in self.manifest.parent.glob('sample_*.png'):
            self.assertEqual(image.read_bytes(), (other.parent / image.name).read_bytes())
        changed = generate(self.root / 'changed', seed=43, replicates=1)
        self.assertNotEqual((changed.parent / 'coordinates.jsonl').read_bytes(),
                            (self.manifest.parent / 'coordinates.jsonl').read_bytes())
        twice = generate(self.root / 'twice', seed=42, replicates=2)
        rows = [json.loads(s) for s in (twice.parent / 'coordinates.jsonl').read_text().splitlines()]
        self.assertEqual(rows[:68], self.rows)
        self.assertNotEqual(rows[0]['coordinates'], rows[68]['coordinates'])
        with self.assertRaises(FileExistsError):
            generate(other.parent)
        with self.assertRaises(ValueError):
            generate(self.root / 'invalid', replicates=0)
        self.assertFalse((self.root / 'invalid').exists())

    def test_coverage_and_geometry(self):
        self.assertEqual(len(self.rows), 68)
        self.assertEqual({r['stress']['family'] for r in self.rows},
                         {'baseline', 'density', 'noise', 'rotation', 'translation',
                          'sparse', 'partial_ring', 'mixed', 'blank'})
        by_id = {r['id']: r for r in self.rows}
        for row in self.rows:
            points, s = row['coordinates'], row['stress']
            self.assertEqual(len(points), s['point_count'])
            self.assertTrue(all(math.isfinite(x) and math.isfinite(y) and x*x+y*y <= 1+1e-12 for x,y in points))
            if s['family'] in ('partial_ring', 'mixed', 'blank', 'sparse'):
                self.assertIsNone(row['label'])
                self.assertFalse(s['semantic_preserving'])
            if s['family'] == 'rotation':
                source = by_id[s['reference_id']]['coordinates']
                self.assertEqual(len(points), len(source))
                a = math.radians(s['parameters']['degrees'])
                for (x, y), (u, v) in zip(source, points):
                    self.assertAlmostEqual(u, x*math.cos(a)-y*math.sin(a))
                    self.assertAlmostEqual(v, x*math.sin(a)+y*math.cos(a))
            if s['family'] == 'density':
                original = by_id[s['reference_id']]['coordinates']
                self.assertEqual(len(points), len(original)//2 if s['level'] == 'half' else len(original)*2)
            if s['family'] == 'noise':
                original = by_id[s['reference_id']]['coordinates']
                self.assertEqual(points[:len(original)], original)
                self.assertEqual(len(points)-len(original), s['parameters']['added_count'])
            if s['family'] == 'partial_ring':
                p = s['parameters']
                for x, y in points[:p['structured_count']]:
                    self.assertTrue(.82 <= math.hypot(x, y) <= .94)
                    a = (math.atan2(y, x)-p['start_radians']) % (2*math.pi)
                    self.assertLessEqual(a, math.radians(p['arc_degrees']) + 1e-12)
            if s['family'] == 'translation':
                self.assertEqual(len(points)+s['parameters']['dropped_points'], s['parameters']['source_count'])
                if s['parameters']['dropped_points'] or s['level'] == 'x_0.4':
                    self.assertIsNone(row['label'])

    def test_manifest_opt_in_and_image_coordinate_equivalence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                load_manifest(self.manifest, self.labels, root / 'reject')
            a = load_manifest(self.manifest, self.labels, root / 'images', stress_suite=True)
            b = load_manifest(self.manifest.parent / 'coordinates.jsonl', self.labels,
                              root / 'coords', self.cfg['render'], stress_suite=True)
            self.assertEqual([r['image_sha256'] for r in a], [r['image_sha256'] for r in b])
            self.assertEqual([r['stress'] for r in a], [r['stress'] for r in b])
            validate_suite(a, self.cfg)
            validate_suite(b, self.cfg)
            bad = root / 'bad.jsonl'
            for patch_value in ({'label': 'not_a_class'}, {'label': None, 'stress': None}):
                row = dict(json.loads(self.manifest.read_text().splitlines()[0]), **patch_value)
                row['image'] = str(self.manifest.parent / row['image'])
                bad.write_text(json.dumps(row)+'\n')
                if row['label'] is not None:
                    with self.assertRaises(ValueError):
                        load_manifest(bad, self.labels, root / 'bad', stress_suite=True)
                else:
                    loaded = load_manifest(bad, self.labels, root / 'bad2', stress_suite=True)
                    with self.assertRaises(ValueError):
                        validate_suite(loaded, self.cfg)

    def test_semantic_and_pair_validation(self):
        cases = [('reference_id', 'missing'), ('source_seed', -1), ('seed', True),
                 ('semantic_preserving', 'yes'), ('evaluation', 'exploratory'),
                 ('config_sha256', 'wrong'), ('replicate', 999)]
        for key, value in cases:
            rows = copy.deepcopy(self.rows)
            rows[1]['stress'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_suite(rows, self.cfg)
        rows = copy.deepcopy(self.rows)
        row = next(r for r in rows if r['stress']['family'] == 'mixed')
        row['label'] = row['stress']['source_class']
        row['stress'].update(evaluation='labeled', semantic_preserving=True)
        with self.assertRaises(ValueError):
            validate_suite(rows, self.cfg)
        rows = copy.deepcopy(self.rows)
        rows[1]['coordinates'][0][0] += .01
        with self.assertRaises(ValueError):
            validate_suite(rows, self.cfg)
        changed_config = copy.deepcopy(self.cfg)
        changed_config['rules'].append('Changed taxonomy')
        with self.assertRaises(ValueError):
            validate_suite(self.rows, changed_config)

    def test_consistency_denominators_and_wrong_constant_predictions(self):
        predictions = [dict(r, prediction='uniform') for r in self.rows]
        report = summarize(predictions, self.labels)
        self.assertEqual(report['overall']['consistency']['consistency'], 1)
        self.assertLess(report['overall']['labeled_metrics']['accuracy'], 1)
        # Two invalid answers must not count as stable successful predictions.
        for r in predictions:
            r['prediction'] = None
        report = summarize(predictions, self.labels)
        summary = report['overall']['consistency']
        self.assertEqual(summary['consistency'], 0)
        self.assertEqual(summary['valid_pair_coverage'], 0)
        self.assertIsNone(summary['consistency_given_both_valid'])
        self.assertIsNone(summary['retained_accuracy_given_correct_reference'])
        self.assertTrue(all(not p['consistent'] for p in report['pairs']))
        # Known 3-pair example: unchanged, changed, invalid.
        rows = [copy.deepcopy(self.rows[i]) for i in (0, 1, 2, 3)]
        for row, pred in zip(rows, ('uniform', 'uniform', 'scratch', None)):
            row['prediction'] = pred
        summary = summarize(rows, self.labels)['overall']['consistency']
        self.assertEqual(summary['pairs'], 3)
        self.assertEqual(summary['consistency'], 1/3)
        self.assertEqual(summary['valid_pair_coverage'], 2/3)
        self.assertEqual(summary['consistency_given_both_valid'], 1/2)
        self.assertEqual(summary['retained_accuracy_given_correct_reference'], 1/3)
        random.Random(1).shuffle(rows)
        self.assertEqual(summarize(rows, self.labels)['overall']['consistency'], summary)

    def test_exploratory_outputs_do_not_affect_accuracy(self):
        rows = [dict(r, prediction=r['label'] if r['label'] else 'scratch') for r in self.rows]
        report = summarize(rows, self.labels)
        labeled = report['overall']['labeled_metrics']
        self.assertEqual(labeled['accuracy'], 1)
        self.assertEqual(labeled['n'], sum(r['label'] is not None for r in rows))
        self.assertIsNone(report['by_family']['mixed']['labeled_metrics'])
        self.assertEqual(sum(report['by_family']['mixed']['exploratory_prediction_counts'].values()), 9)
        for r in rows:
            if r['label'] is None:
                r['prediction'] = None
        self.assertEqual(summarize(rows, self.labels)['overall']['labeled_metrics'], labeled)
        self.assertEqual(summarize(rows, self.labels)['by_family']['mixed']['exploratory_invalid_output_rate'], 1)

    def test_shared_runner_and_failure_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = parser().parse_args(['--config', str(CONFIG), '--manifest', str(self.manifest),
                '--stress-suite', '--backend', 'mock', '--output', str(root / 'run')])
            with contextlib.redirect_stdout(io.StringIO()):
                report = execute(args)
            out = Path(args.output)
            predictions = [json.loads(x) for x in (out / 'predictions.jsonl').read_text().splitlines()]
            exploratory = [json.loads(x) for x in (out / 'exploratory_predictions.jsonl').read_text().splitlines()]
            self.assertEqual(len(predictions), 68)
            self.assertEqual(len(exploratory), report['exploratory_samples'])
            self.assertTrue(all(r['label'] is None and r['stress']['evaluation'] == 'exploratory' for r in exploratory))
            self.assertEqual(report['n'] + len(exploratory), 68)
            stress = json.loads((out / 'stress_metrics.json').read_text())
            self.assertTrue(stress['pipeline_test_only'])
            self.assertFalse(stress['model_quality_evaluation'])
            self.assertEqual(stress['evidence_level'], 'cpu_mock')
            self.assertFalse(stress['real_fab_accuracy_validated'])
            self.assertEqual(json.loads((out / 'run.json').read_text())['status'], 'complete')
            args.output = str(root / 'fail')
            with patch('defect_benchmark.backends.MockBackend.predict', side_effect=RuntimeError('inference failed')):
                with self.assertRaises(RuntimeError):
                    execute(args)
            self.assertEqual(json.loads((Path(args.output) / 'run.json').read_text())['status'], 'failed')
            self.assertFalse((Path(args.output) / 'stress_metrics.json').exists())
            # Broken metadata fails before backend construction.
            broken = json.loads(self.manifest.read_text().splitlines()[0])
            broken['image'] = str(self.manifest.parent / broken['image'])
            broken['stress']['config_sha256'] = 'bad'
            bad = root / 'bad.jsonl'
            bad.write_text(json.dumps(broken)+'\n')
            args.manifest, args.output = str(bad), str(root / 'bad-run')
            with patch.dict('defect_benchmark.run.BACKENDS', mock=lambda _: self.fail('loaded backend')):
                with self.assertRaises(ValueError):
                    execute(args)

    def test_default_smoke_remains_five_labeled_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = generate_smoke(root / 'data', per_class=1)
            args = parser().parse_args(['--config', str(CONFIG), '--manifest', str(manifest),
                '--backend', 'mock', '--output', str(root / 'run')])
            self.assertFalse(args.stress_suite)
            with contextlib.redirect_stdout(io.StringIO()):
                report = execute(args)
            self.assertEqual(report['n'], 5)
            self.assertNotIn('labeled_samples', report)
            self.assertFalse((root / 'run/stress_metrics.json').exists())
            self.assertFalse((root / 'run/exploratory_predictions.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
