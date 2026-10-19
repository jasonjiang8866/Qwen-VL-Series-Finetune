import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.wafer_zero_shot.core import render, metrics, messages, parse_label, validate_config
from benchmarks.wafer_zero_shot.synthetic import generate
from benchmarks.wafer_zero_shot.backends import MockBackend
from benchmarks.wafer_zero_shot.__main__ import evaluate

CONFIG = Path(__file__).parents[1] / 'configs/default.json'


class BenchmarkTests(unittest.TestCase):
    def setUp(self): self.config = validate_config(json.loads(CONFIG.read_text()))

    def test_strict_parser(self):
        for value in ('CENTER RING', 'CENTER.', '{"label":"CENTER"}', 'center', 'The answer is CENTER', '', '```CENTER```'):
            self.assertIsNone(parse_label(value, self.config['labels']))
        self.assertEqual(parse_label(' \nCENTER\t', self.config['labels']), 'CENTER')

    def test_hand_calculated_metrics(self):
        m = metrics(['A', 'A', 'B', 'B'], ['A', None, 'A', 'B'], ['A', 'B', 'C'])
        self.assertEqual(m['accuracy'], .5)
        self.assertEqual(m['invalid_output_rate'], .25)
        self.assertEqual(m['recall'], {'A': .5, 'B': .5, 'C': 0})
        self.assertAlmostEqual(m['macro_f1'], (.5 + 2/3) / 3)
        self.assertEqual(m['confusion_matrix']['counts'], [[1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 0, 0]])

    def test_renderer_frame_orientation_and_invariance(self):
        a = render([[0, .5]], size=100)
        self.assertEqual(a.getpixel((50, 33)), (0, 0, 0))
        self.assertEqual(a.getpixel((50, 66)), (255, 255, 255))
        b = render([[10, 25]], center=[10, 20], radius=10, size=100)
        self.assertEqual(a.tobytes(), b.tobytes())
        points = [[.1, .2], [-.2, -.1]]
        self.assertEqual(render(points).tobytes(), render(points[::-1]).tobytes())

    def test_bad_coordinates(self):
        for points in ([[2, 0]], [[float('nan'), 0]], [[0]], [['x', 0]]):
            with self.assertRaises(ValueError): render(points)

    def test_reproducible_synthetic(self):
        self.assertEqual(generate(self.config), generate(self.config))
        self.assertNotEqual(generate(self.config, seed=1), generate(self.config, seed=2))
        for r in generate(self.config): render(r['coordinates'])

    def test_prompts_and_no_truth_leak(self):
        for strategy in self.config['strategies']:
            prompt = messages(self.config, strategy)
            self.assertEqual(prompt[1]['content'][0], {'type': 'image'})
            self.assertNotIn('synthetic-', str(prompt))
            for label in self.config['labels']: self.assertIn(label, prompt[0]['content'])

    def test_end_to_end_mock(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'run'
            result = evaluate(generate(self.config, 2), self.config, MockBackend('CENTER'), out)
            self.assertEqual(len(result), 3)
            for summary in result.values():
                self.assertEqual(summary['n'], 10)
                self.assertEqual(summary['accuracy'], .2)
                self.assertTrue(summary['pipeline_test_only'])
            self.assertEqual(len(list((out / 'images').glob('*.png'))), 10)
            with self.assertRaises(FileExistsError):
                evaluate(generate(self.config, 2), self.config, MockBackend('CENTER'), out)

    def test_invalid_output_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate(generate(self.config, 1), self.config, MockBackend('CENTER or RING'), Path(directory) / 'run', 0)
            for summary in result.values():
                self.assertEqual(summary['accuracy'], 0)
                self.assertEqual(summary['invalid_output_rate'], 1)


if __name__ == '__main__': unittest.main()

class AdapterContractTests(unittest.TestCase):
    def test_image_text_and_prompt_token_trimming(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        from benchmarks.wafer_zero_shot.backends import TransformersBackend

        calls = {}

        class Inputs(dict):
            def to(self, device): return self

        class Output:
            def __getitem__(self, index):
                calls['trim_start'] = index[1].start
                return 'new_tokens_only'

        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                calls['messages'] = messages
                return 'formatted_prompt'

            def __call__(self, **kwargs):
                calls['processor'] = kwargs
                return Inputs(input_ids=SimpleNamespace(shape=(1, 23)), pixel_values='pixels', token_type_ids='ignored')

            def batch_decode(self, ids, **kwargs):
                assert ids == 'new_tokens_only'
                return ['RING']

        class Model:
            def generate(self, **kwargs):
                calls['generate'] = kwargs
                return Output()

        backend = TransformersBackend.__new__(TransformersBackend)
        backend.processor, backend.model = Processor(), Model()
        backend.torch = SimpleNamespace(inference_mode=nullcontext)
        backend.device, backend.max_new_tokens = 'cpu', 32
        image = render([])
        prompt = messages(validate_config(json.loads(CONFIG.read_text())), 'definitions')
        self.assertEqual(backend.generate(image, prompt), 'RING')
        self.assertIs(calls['processor']['images'][0], image)
        self.assertEqual(calls['trim_start'], 23)
        self.assertEqual(calls['generate']['pixel_values'], 'pixels')
        self.assertFalse(calls['generate']['do_sample'])
        self.assertNotIn('token_type_ids', calls['generate'])
