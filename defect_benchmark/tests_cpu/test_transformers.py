"""Real pinned library CPU contracts, no checkpoint weights or CUDA inference.

Run separately from the dependency-light suite. Missing dependencies are errors,
not skipped tests. Image processor configs use the official checkpoints' geometry.
"""
import contextlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import transformers as tr
from PIL import Image
from transformers.generation.utils import GenerationMixin

from defect_benchmark.backends import TransformersBackend
from defect_benchmark.core import LabelConstraint
from defect_benchmark.run import parser


class TransformersCPUTests(unittest.TestCase):
    def test_real_image_processors_through_backend_initialization(self):
        for model_type, cls_name, patch_size, processor_cls in (
            ('qwen2_5_vl', 'Qwen2_5_VLForConditionalGeneration', 14, tr.Qwen2VLImageProcessor),
            ('qwen2_5_vl', 'Qwen2_5_VLForConditionalGeneration', 14, tr.Qwen2VLImageProcessorFast),
            ('qwen3_vl', 'Qwen3VLForConditionalGeneration', 16, tr.Qwen2VLImageProcessorFast),
        ):
            with self.subTest(model_type=model_type, processor=processor_cls.__name__):
                image_processor = processor_cls(patch_size=patch_size, merge_size=2, temporal_patch_size=2)
                processor = SimpleNamespace(image_processor=image_processor, chat_template='test')
                args = parser().parse_args(['--config', 'unused', '--manifest', 'unused', '--output', 'unused'])
                cfg = SimpleNamespace(model_type=model_type, _commit_hash=None)
                # Only CUDA admission and checkpoint loading are replaced. The real
                # library image processor is initialized and preprocesses CPU pixels.
                with patch.object(torch.cuda, 'is_available', return_value=True), \
                     patch.object(torch.cuda, 'set_device'), \
                     patch.object(torch.cuda, 'is_bf16_supported', return_value=True), \
                     patch.object(torch.cuda, 'get_device_name', return_value='CPU-test-double'), \
                     patch.object(tr.AutoConfig, 'from_pretrained', return_value=cfg), \
                     patch.object(getattr(tr, cls_name), 'from_pretrained', return_value=Mock()), \
                     patch.object(tr.AutoProcessor, 'from_pretrained', return_value=processor):
                    backend = TransformersBackend(args)
                self.assertEqual(image_processor.patch_size, patch_size)
                self.assertEqual(image_processor.size, {'shortest_edge': 65536, 'longest_edge': 262144})
                for dimensions in ((32, 32), (1024, 1024), (2048, 256)):
                    inputs = image_processor(images=[Image.new('RGB', dimensions)], return_tensors='pt')
                    t, h, w = inputs['image_grid_thw'][0].tolist()
                    self.assertEqual(t, 1)
                    self.assertEqual(h % 2, 0)
                    self.assertEqual(w % 2, 0)
                    pixels = h * w * patch_size**2
                    # smart_resize rounds to merge-sized patches, so use one-patch tolerance.
                    self.assertLessEqual(pixels, args.max_pixels + (patch_size*2)**2)
                    self.assertGreaterEqual(pixels, args.min_pixels - (patch_size*2)**2)
                    self.assertEqual(inputs['pixel_values'].shape, (t*h*w, 3*2*patch_size**2))
                    self.assertTrue(torch.isfinite(inputs['pixel_values']).all())
                self.assertEqual(backend.metadata['image_processor']['patch_size'], patch_size)

    def test_actual_generation_config_merge_stays_greedy(self):
        # Reproduce Qwen3's >=4.50 checkpoint defaults with the real 4.57.1
        # merge implementation. Exercise the adapter call, not just a copied flag.
        from transformers import GenerationConfig
        class Tokenizer:
            eos_token_id, pad_token_id = 1, 0
            def encode(self, value, **kwargs): return [2] if value == 'a' else [2, 3]
            def decode(self, ids, **kwargs):
                return ''.join({2:'a', 3:'b'}.get(int(x), '') for x in ids)
        class Processor:
            tokenizer = Tokenizer()
            def apply_chat_template(self, *args, **kwargs): return 'prompt'
            def __call__(self, **kwargs):
                from transformers.feature_extraction_utils import BatchFeature
                return BatchFeature({'input_ids': torch.tensor([[9, 9]]),
                    'image_grid_thw': torch.tensor([[1, 8, 8]])})
        class Model:
            generation_config = GenerationConfig(do_sample=True, top_k=20, top_p=.8, temperature=.7,
                repetition_penalty=1.05, transformers_version='4.56.0')
            def generate(self, input_ids, image_grid_thw, generation_config, **kwargs):
                self.effective, _ = GenerationMixin._prepare_generation_config(
                    self, generation_config, use_model_defaults=kwargs.get('use_model_defaults'))
                self.constraint = kwargs.get('prefix_allowed_tokens_fn')
                return torch.tensor([[9, 9, 2, 1]])
        model = Model()
        # Establish the regression mechanism, which fake GenerationConfig tests miss.
        inherited, _ = GenerationMixin._prepare_generation_config(model, GenerationConfig(do_sample=False))
        self.assertTrue(inherited.do_sample)
        backend = TransformersBackend.__new__(TransformersBackend)
        backend.torch, backend.device = torch, torch.device('cpu')
        backend.model, backend.processor = model, Processor()
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(torch.cuda, 'synchronize'), \
             patch.object(torch.cuda, 'reset_peak_memory_stats'), \
             patch.object(torch.cuda, 'max_memory_allocated', return_value=0), \
             patch.object(torch.cuda, 'max_memory_reserved', return_value=0):
            path = Path(tmp) / 'image.png'
            Image.new('RGB', (32, 32)).save(path)
            for mode in ('strict', 'constrained'):
                backend.args = SimpleNamespace(output_mode=mode, max_new_tokens=4)
                output = backend.predict(path, 'system', ['a', 'ab'])
                self.assertFalse(model.effective.do_sample)
                self.assertEqual(model.effective.num_beams, 1)
                self.assertEqual(model.effective.repetition_penalty, 1.0)
                self.assertEqual(output['raw_output'], 'a')
                self.assertFalse(output['truncated'])
                if mode == 'constrained':
                    self.assertIsInstance(model.constraint, LabelConstraint)
                    self.assertEqual(model.constraint(0, torch.tensor([9, 9, 2])), [1, 3])
                else:
                    self.assertIsNone(model.constraint)


if __name__ == '__main__':
    unittest.main()
