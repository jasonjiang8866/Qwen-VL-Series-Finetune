import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from defect_benchmark.core import LabelConstraint, build_prompt, load_config, load_manifest, metrics, parse_output, sha256
from defect_benchmark.run import execute, parser
from defect_benchmark.synthetic import generate

CONFIG = Path(__file__).resolve().parents[1] / 'configs/example.json'

class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())
    def decode(self, ids, **kwargs):
        return bytes(ids).decode()

class BenchmarkTests(unittest.TestCase):
    def test_strict_output(self):
        for raw in ['ring scratch', '{"class":"ring"}', 'RING', 'ring.', '```ring```', '', 'ring\nexplanation']:
            self.assertIsNone(parse_output(raw, ['ring', 'scratch']))
        self.assertEqual(parse_output(' ring\n', ['ring']), 'ring')

    def test_metrics_invalid_is_false_negative(self):
        m = metrics(['a', 'a', 'b', 'b'], ['a', None, 'a', 'b'], ['a', 'b', 'absent'])
        self.assertEqual(m['accuracy'], .5)
        self.assertEqual(m['invalid_output_rate'], .25)
        self.assertAlmostEqual(m['macro_f1'], (.5 + 2/3) / 3)
        self.assertEqual(m['confusion_matrix']['counts'], [[1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(metrics(['a'], [None], ['a'])['macro_f1'], 0)
        with self.assertRaises(ValueError):
            metrics([], [], ['a'])

    def test_token_prefix_overlap_and_eos(self):
        c = LabelConstraint(Tokenizer(), ['a', 'ab', 'b'], 10, 0)
        self.assertEqual(c.allowed([]), [97, 98])
        self.assertEqual(c.allowed([97]), [0, 98])
        self.assertEqual(c.allowed([97, 98]), [0])
        with self.assertRaises(ValueError):
            c.allowed([99])
        self.assertEqual(c.max_tokens, 3)

    def test_prompt_and_config(self):
        cfg = load_config(CONFIG)
        prompt = build_prompt(cfg)
        self.assertIn('NOT a Wafer Bin Map', prompt)
        for cls in cfg['classes']:
            self.assertIn(cls['description'], prompt)
        self.assertNotIn('sample_', prompt)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.json'
            cfg['classes'].append(cfg['classes'][0])
            path.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_synthetic_repeatability_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = generate(Path(tmp) / 'a'), generate(Path(tmp) / 'b')
            self.assertEqual(sha256(a), sha256(b))
            self.assertEqual(sha256(a.parent / 'sample_000.png'), sha256(b.parent / 'sample_000.png'))
            labels = [x['id'] for x in load_config(CONFIG)['classes']]
            rows = load_manifest(a, labels)
            self.assertEqual(len(rows), 10)
            self.assertTrue(all(Path(x['image']).is_absolute() for x in rows))
            text = a.read_text()
            a.write_text(text + text.splitlines()[0] + '\n')
            with self.assertRaises(ValueError):
                load_manifest(a, labels)
            a.write_text(text.replace('"uniform"', '"unknown"', 1))
            with self.assertRaises(ValueError):
                load_manifest(a, labels)
            a.write_text(text)
            (a.parent / 'sample_000.png').write_text('corrupt')
            with self.assertRaises(OSError):
                load_manifest(a, labels)

    def test_end_to_end_mock(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, output = generate(Path(tmp) / 'data'), Path(tmp) / 'run'
            args = parser().parse_args(['--config', str(CONFIG), '--manifest', str(manifest), '--backend', 'mock', '--output', str(output)])
            with contextlib.redirect_stdout(io.StringIO()):
                result = execute(args)
            self.assertFalse(result['model_quality_evaluation'])
            self.assertIsNone(result['latency_s'])
            self.assertEqual(result['n'], 10)
            self.assertEqual(json.loads((output / 'run.json').read_text())['status'], 'complete')
            self.assertEqual(len((output / 'predictions.jsonl').read_text().splitlines()), 10)
            with self.assertRaises(FileExistsError):
                execute(args)

    def test_runtime_failure_is_not_accuracy(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, output = generate(Path(tmp) / 'data', per_class=1), Path(tmp) / 'run'
            args = parser().parse_args(['--config', str(CONFIG), '--manifest', str(manifest), '--backend', 'mock', '--warmup', '0', '--output', str(output)])
            with patch('defect_benchmark.backends.MockBackend.predict', side_effect=RuntimeError('simulated OOM')):
                with self.assertRaises(RuntimeError):
                    execute(args)
            self.assertFalse((output / 'metrics.json').exists())
            self.assertEqual(json.loads((output / 'run.json').read_text())['status'], 'failed')

if __name__ == '__main__':
    unittest.main()
