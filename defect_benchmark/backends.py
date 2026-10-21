"""Backend contract: predict(image_path, system_prompt, labels) -> measurement dict.
Backends never receive ground truth or sample IDs. Add adapters in BACKENDS.
"""
import time
from .core import LabelConstraint


class MockBackend:
    """Plumbing only: fixed answer independent of image and truth."""
    def __init__(self, args):
        self.metadata = {'backend': 'mock', 'model_quality_evaluation': False}

    def predict(self, image_path, system_prompt, labels):
        from PIL import Image
        with Image.open(image_path) as image:
            image.convert('RGB').load()
        return dict(raw_output=labels[0], latency_s=None, generation_s=None,
                    peak_allocated_bytes=None, peak_reserved_bytes=None)


class TransformersBackend:
    def __init__(self, args):
        import torch
        import transformers as tr
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU required for transformers backend; use mock for CPU smoke test')
        self.torch, self.args = torch, args
        self.device = torch.device(args.device)
        torch.cuda.set_device(self.device)
        if args.dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
            raise ValueError('GPU does not support bf16; use --dtype float16')
        tr.set_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        dtype = getattr(torch, args.dtype)
        common = dict(revision=args.revision, local_files_only=args.local_files_only, trust_remote_code=False)
        cfg = tr.AutoConfig.from_pretrained(args.model, **common)
        # Resolve mutable refs once so weights/tokenizer/processor cannot drift
        # from the config revision recorded in the results.
        resolved_revision = getattr(cfg, '_commit_hash', None)
        if resolved_revision is not None:
            common['revision'] = resolved_revision
        names = {'qwen2_5_vl': 'Qwen2_5_VLForConditionalGeneration', 'qwen3_vl': 'Qwen3VLForConditionalGeneration'}
        if cfg.model_type not in names:
            raise ValueError(f'Unsupported {cfg.model_type}; add a backend adapter')
        cls = getattr(tr, names[cfg.model_type], None)
        if cls is None:
            raise RuntimeError('Install benchmark requirements (Transformers 4.57.1)')
        kwargs = dict(dtype=dtype, device_map={'': str(self.device)}, attn_implementation='sdpa')
        if args.bits != 16:
            kwargs['quantization_config'] = tr.BitsAndBytesConfig(
                load_in_4bit=args.bits == 4, load_in_8bit=args.bits == 8,
                bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)
        load_start = time.perf_counter()
        self.model = cls.from_pretrained(args.model, **common, **kwargs).eval()
        self.processor = tr.AutoProcessor.from_pretrained(args.model, **common)
        # Both processors expose size budgets; Qwen3 uses a different patch size.
        self.processor.image_processor.size['shortest_edge'] = args.min_pixels
        self.processor.image_processor.size['longest_edge'] = args.max_pixels
        # Slow Qwen2 image processors also keep legacy pixel-budget attributes.
        if hasattr(self.processor.image_processor, 'min_pixels'):
            self.processor.image_processor.min_pixels = args.min_pixels
            self.processor.image_processor.max_pixels = args.max_pixels
        torch.cuda.synchronize(self.device)
        quantized = [m for m in self.model.modules() if type(m).__name__ == 'Linear4bit']
        self.metadata = dict(backend='transformers',
                             model_load_and_processor_s=time.perf_counter() - load_start,
                             loaded_in_4bit=getattr(self.model, 'is_loaded_in_4bit', False) is True,
                             linear4bit_modules=len(quantized),
                             quantized_compute_dtypes=sorted({str(m.compute_dtype) for m in quantized}),
                             parameter_dtypes=sorted({str(p.dtype) for p in self.model.parameters()}),
                             parameter_devices=sorted({str(p.device) for p in self.model.parameters()}),
                             model_quality_evaluation=True,
                             model_type=cfg.model_type, resolved_model_revision=getattr(cfg, '_commit_hash', None),
                             gpu=torch.cuda.get_device_name(self.device), cuda=torch.version.cuda,
                             image_processor=self.processor.image_processor.to_dict(),
                             chat_template=self.processor.chat_template)

    def predict(self, image_path, system_prompt, labels):
        from PIL import Image
        from transformers import GenerationConfig
        torch, args = self.torch, self.args
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        with Image.open(image_path) as source:
            image = source.convert('RGB')
        messages = [dict(role='system', content=system_prompt),
                    dict(role='user', content=[{'type': 'image'}, {'type': 'text', 'text': 'Classify this defect distribution.'}])]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors='pt').to(self.device)
        inputs.pop('token_type_ids', None)
        length = inputs['input_ids'].shape[1]
        tokenizer = self.processor.tokenizer
        eos = tokenizer.eos_token_id
        constraint = None
        if args.output_mode == 'constrained':
            constraint = LabelConstraint(tokenizer, labels, length, eos)
            if args.max_new_tokens < constraint.max_tokens:
                raise ValueError(f'--max-new-tokens must be >= {constraint.max_tokens}')
        generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens,
                                      eos_token_id=eos, pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos,
                                      use_cache=True)
        extra = {'prefix_allowed_tokens_fn': constraint} if args.output_mode == 'constrained' else {}
        torch.cuda.synchronize(self.device)
        generation_start = time.perf_counter()
        with torch.inference_mode():
            # In Transformers >=4.50 global-default values in GenerationConfig
            # can be replaced by checkpoint defaults (including do_sample=True).
            output = self.model.generate(**inputs, generation_config=generation,
                                         use_model_defaults=False, **extra)
        torch.cuda.synchronize(self.device)
        generation_s = time.perf_counter() - generation_start
        generated = output[0, length:]
        raw = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        terminated = bool(len(generated) and generated[-1].item() == eos)
        return dict(raw_output=raw, latency_s=time.perf_counter() - start, generation_s=generation_s,
                    finish_reason='eos' if terminated else 'length', truncated=not terminated,
                    generated_token_ids=[int(token) for token in generated], eos_token_id=eos,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device),
                    input_tokens=length, output_tokens=output.shape[1] - length,
                    image_grid_thw=inputs['image_grid_thw'].tolist())


BACKENDS = {'mock': MockBackend, 'transformers': TransformersBackend}
