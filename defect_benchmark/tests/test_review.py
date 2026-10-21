"""Offline regressions for review findings; CUDA/model loads are test doubles."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from defect_benchmark.backends import TransformersBackend
from defect_benchmark.run import execute, parser
from defect_benchmark.synthetic import generate

CONFIG = Path(__file__).resolve().parents[1] / 'configs/example.json'


class ReviewTests(unittest.TestCase):
    def test_constructor_dispatch_budgets_and_revision(self):
        for model_type, model_class, patch_size in (
            ('qwen2_5_vl', 'Qwen2_5_VLForConditionalGeneration', 14),
            ('qwen3_vl', 'Qwen3VLForConditionalGeneration', 16),
        ):
            for bits in (16, 8, 4):
                with self.subTest(model_type=model_type, bits=bits):
                    args = parser().parse_args(['--config', str(CONFIG), '--manifest', 'unused',
                        '--output', 'unused', '--bits', str(bits), '--local-files-only'])
                    image_processor = SimpleNamespace(size={'shortest_edge': 3136, 'longest_edge': 999999},
                        min_pixels=3136, max_pixels=999999, patch_size=patch_size, to_dict=lambda: {})
                    processor = SimpleNamespace(image_processor=image_processor, chat_template='template')
                    loader = Mock()
                    tr = SimpleNamespace(AutoConfig=Mock(), AutoProcessor=Mock(),
                        BitsAndBytesConfig=Mock(), set_seed=Mock(), **{model_class: loader})
                    tr.AutoConfig.from_pretrained.return_value = SimpleNamespace(model_type=model_type, _commit_hash='a'*40)
                    tr.AutoProcessor.from_pretrained.return_value = processor
                    torch = SimpleNamespace(cuda=Mock(), device=lambda x: x, bfloat16='bf16',
                        version=SimpleNamespace(cuda='test-double'), backends=SimpleNamespace(
                            cudnn=SimpleNamespace(benchmark=True), cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))))
                    torch.cuda.get_device_name.return_value = 'test-double'
                    with patch.dict('sys.modules', torch=torch, transformers=tr):
                        backend = TransformersBackend(args)
                    self.assertEqual(image_processor.size, {'shortest_edge': 65536, 'longest_edge': 262144})
                    self.assertEqual((image_processor.min_pixels, image_processor.max_pixels), (65536, 262144))
                    self.assertEqual(image_processor.patch_size, patch_size)
                    self.assertEqual(backend.metadata['resolved_model_revision'], 'a'*40)
                    self.assertEqual(tr.AutoConfig.from_pretrained.call_args.kwargs['revision'], 'main')
                    for call in (loader.from_pretrained.call_args, tr.AutoProcessor.from_pretrained.call_args):
                        self.assertEqual(call.kwargs['revision'], 'a'*40)
                        self.assertTrue(call.kwargs['local_files_only'])
                        self.assertFalse(call.kwargs['trust_remote_code'])
                    kwargs = loader.from_pretrained.call_args.kwargs
                    self.assertEqual(kwargs['device_map'], {'': 'cuda:0'})
                    self.assertEqual(kwargs['attn_implementation'], 'sdpa')
                    if bits == 16:
                        self.assertNotIn('quantization_config', kwargs)
                    else:
                        quant = tr.BitsAndBytesConfig.call_args.kwargs
                        self.assertEqual(quant['load_in_4bit'], bits == 4)
                        self.assertEqual(quant['load_in_8bit'], bits == 8)

    def test_truncated_valid_label_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = generate(Path(tmp) / 'data', per_class=1)
            output = Path(tmp) / 'run'
            args = parser().parse_args(['--config', str(CONFIG), '--manifest', str(manifest),
                '--output', str(output), '--backend', 'mock', '--warmup', '0'])
            result = dict(raw_output='center_cluster', truncated=True, finish_reason='length',
                latency_s=None, generation_s=None, peak_allocated_bytes=None, peak_reserved_bytes=None)
            with patch('defect_benchmark.backends.MockBackend.predict', return_value=result), contextlib.redirect_stdout(io.StringIO()):
                report = execute(args)
            self.assertEqual(report['accuracy'], 0)
            self.assertEqual(report['invalid_output_rate'], 1)
            records = [json.loads(x) for x in (output / 'predictions.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['prediction'] is None and r['raw_output'] == 'center_cluster' for r in records))

    def test_partial_failure_preserves_records_without_metrics(self):
        from defect_benchmark.backends import MockBackend
        original = MockBackend.predict
        for phase in ('initialization', 'warmup', 'measured'):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as tmp:
                manifest = generate(Path(tmp) / 'data', per_class=1)
                output = Path(tmp) / 'run'
                args = parser().parse_args(['--config', str(CONFIG), '--manifest', str(manifest),
                    '--output', str(output), '--backend', 'mock', '--warmup', '1' if phase == 'warmup' else '0'])
                count = 0
                def predict(self, *params):
                    nonlocal count
                    count += 1
                    if phase == 'warmup' or count == 2:
                        raise RuntimeError('simulated failure')
                    return original(self, *params)
                target = 'defect_benchmark.backends.MockBackend.' + ('__init__' if phase == 'initialization' else 'predict')
                replacement = Mock(side_effect=RuntimeError('simulated failure')) if phase == 'initialization' else predict
                with patch(target, replacement), self.assertRaises(RuntimeError):
                    execute(args)
                metadata = json.loads((output / 'run.json').read_text())
                self.assertEqual(metadata['status'], 'failed')
                self.assertEqual(metadata['completed_samples'], 1 if phase == 'measured' else 0)
                self.assertFalse((output / 'metrics.json').exists())
                if phase == 'measured':
                    self.assertEqual(len((output / 'predictions.jsonl').read_text().splitlines()), 1)


if __name__ == '__main__':
    unittest.main()
