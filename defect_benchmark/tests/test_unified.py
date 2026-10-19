"""CPU regression checks; fake tensors do not validate CUDA/model compatibility."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from defect_benchmark.core import STRATEGIES, build_prompt, load_config, load_manifest
from defect_benchmark.render import render
from defect_benchmark.run import execute, parser
from defect_benchmark.synthetic import generate

CONFIG = Path(__file__).resolve().parents[1] / 'configs/example.json'

class UnifiedTests(unittest.TestCase):
    def test_geometry_orientation_units_and_order(self):
        a = render([[0, .5]], size=100)
        self.assertEqual(a.getpixel((50, 33)), (0, 0, 0))
        self.assertEqual(a.getpixel((50, 66)), (255, 255, 255))
        self.assertEqual(a.tobytes(), render([[10, 25]], center=[10, 20], radius=10, size=100).tobytes())
        points = [[.1, .2], [-.2, -.1]]
        self.assertEqual(render(points).tobytes(), render(points[::-1]).tobytes())
        for points in ([[2, 0]], [[float('nan'), 0]], [[0]], [['x', 0]]):
            with self.assertRaises(ValueError):
                render(points)

    def test_prompt_ablation_keeps_taxonomy(self):
        cfg = load_config(CONFIG)
        prompts = [build_prompt(cfg, x) for x in STRATEGIES]
        self.assertEqual(len(set(prompts)), 3)
        for prompt in prompts:
            for cls in cfg['classes']:
                self.assertIn(cls['description'], prompt)
            for rule in cfg['rules']:
                self.assertIn(rule, prompt)
        with self.assertRaises(ValueError):
            build_prompt(cfg, 'typo')

    def test_coordinates_and_images_have_identical_model_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = generate(Path(tmp) / 'data', per_class=1)
            cfg = load_config(CONFIG)
            labels = [c['id'] for c in cfg['classes']]
            a = load_manifest(manifest, labels, Path(tmp) / 'a')
            b = load_manifest(manifest.parent / 'coordinates.jsonl', labels, Path(tmp) / 'b', cfg['render'])
            self.assertEqual([r['image_sha256'] for r in a], [r['image_sha256'] for r in b])
            self.assertTrue(all(r['synthetic'] for r in a+b))
            rows = [json.loads(x) for x in manifest.read_text().splitlines()]
            rows[0]['coordinates'] = []
            manifest.write_text('\n'.join(json.dumps(x) for x in rows))
            with self.assertRaises(ValueError):
                load_manifest(manifest, labels, Path(tmp) / 'bad')

    def test_six_cell_matrix_and_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = generate(Path(tmp) / 'data', per_class=1)
            for strategy in STRATEGIES:
                for mode in ('strict', 'constrained'):
                    out = Path(tmp) / f'{strategy}-{mode}'
                    args = parser().parse_args(['--config', str(CONFIG), '--manifest', str(manifest),
                        '--backend', 'mock', '--output', str(out), '--prompt-strategy', strategy, '--output-mode', mode])
                    with contextlib.redirect_stdout(io.StringIO()):
                        report = execute(args)
                    self.assertEqual(report['evidence_level'], 'cpu_mock')
                    self.assertTrue(report['synthetic_only'])
                    self.assertFalse(report['real_fab_accuracy_validated'])
                    meta = json.loads((out / 'run.json').read_text())
                    self.assertEqual(len(meta['prompt_sha256']), 64)
                    self.assertEqual(len(list((out / 'inputs').glob('*.png'))), 5)

    def test_adapter_strict_independent_and_constrained_contract(self):
        from defect_benchmark.backends import TransformersBackend
        from test_benchmark import Tokenizer
        calls = {}
        class Inputs(dict):
            def to(self, device): return self
        class Output:
            shape = (1, 24)
            def __getitem__(self, index):
                calls['trim_start'] = index[1].start
                return [97]
        class Processor:
            tokenizer = Tokenizer()
            tokenizer.eos_token_id = 1
            tokenizer.pad_token_id = 0
            def apply_chat_template(self, messages, **kwargs):
                calls['messages'] = messages
                return 'prompt'
            def __call__(self, **kwargs):
                calls['processor'] = kwargs
                return Inputs(input_ids=SimpleNamespace(shape=(1, 23)), pixel_values='pixels',
                              token_type_ids='remove', image_grid_thw=SimpleNamespace(tolist=lambda: [[1, 8, 8]]))
        class Model:
            def generate(self, **kwargs):
                calls['generate'] = kwargs
                return Output()
        cuda = SimpleNamespace(synchronize=lambda _: None, reset_peak_memory_stats=lambda _: None,
                               max_memory_allocated=lambda _: 123, max_memory_reserved=lambda _: 456)
        backend = TransformersBackend.__new__(TransformersBackend)
        backend.processor, backend.model = Processor(), Model()
        backend.torch = SimpleNamespace(cuda=cuda, inference_mode=contextlib.nullcontext)
        backend.device = 'cuda:0'
        backend.args = SimpleNamespace(output_mode='strict', max_new_tokens=8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'image.png'
            render([]).save(path)
            with patch.dict('sys.modules', transformers=SimpleNamespace(GenerationConfig=lambda **kw: SimpleNamespace(**kw))):
                with patch('defect_benchmark.backends.LabelConstraint', side_effect=AssertionError('strict must not build trie')):
                    result = backend.predict(path, 'system', ['a'])
                self.assertEqual(result['raw_output'], 'a')
                self.assertEqual(calls['trim_start'], 23)
                self.assertEqual(calls['processor']['images'][0].mode, 'RGB')
                self.assertNotIn('token_type_ids', calls['generate'])
                self.assertNotIn('prefix_allowed_tokens_fn', calls['generate'])
                self.assertEqual(calls['generate']['generation_config'].pad_token_id, 0)
                backend.args.output_mode = 'constrained'
                backend.predict(path, 'system', ['a'])
                self.assertIn('prefix_allowed_tokens_fn', calls['generate'])
                self.assertFalse(calls['generate']['generation_config'].do_sample)
                backend.args.max_new_tokens = 1
                with self.assertRaises(ValueError):
                    backend.predict(path, 'system', ['a'])

if __name__ == '__main__':
    unittest.main()
