"""Backend contract: generate(PIL.Image, messages) -> raw string; no truth input."""
from typing import Protocol


class Backend(Protocol):
    def generate(self, image, messages) -> str: ...
    def synchronize(self): ...
    def reset_peak(self): ...
    def peak_memory(self) -> dict: ...


class MockBackend:
    """Deliberately constant output: tests plumbing, never model quality."""
    def __init__(self, label):
        self.label = label

    def generate(self, image, messages):
        if image.mode != 'RGB' or messages[1]['content'][0]['type'] != 'image':
            raise ValueError('Missing image input')
        return self.label

    def synchronize(self): pass
    def reset_peak(self): pass
    def peak_memory(self): return {}


class TransformersBackend:
    def __init__(self, model_path, dtype='bfloat16', device='cuda:0', max_new_tokens=32):
        import torch
        from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        if config.model_type not in ('qwen2_5_vl', 'qwen3_vl', 'qwen3_vl_moe'):
            raise ValueError('Unsupported model family; implement a separate Backend adapter')
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
            torch_dtype=getattr(torch, dtype), attn_implementation='sdpa').to(self.device).eval()
        self.max_new_tokens = max_new_tokens

    def generate(self, image, messages):
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image], return_tensors='pt', padding=True).to(self.device)
        inputs.pop('token_type_ids', None)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=self.max_new_tokens)
        # Decode ONLY new tokens; the prompt itself contains every valid label.
        return self.processor.batch_decode(output[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def synchronize(self):
        if self.device.type == 'cuda': self.torch.cuda.synchronize(self.device)

    def reset_peak(self):
        if self.device.type == 'cuda': self.torch.cuda.reset_peak_memory_stats(self.device)

    def peak_memory(self):
        if self.device.type != 'cuda': return {}
        return {'device': str(self.device), 'allocated_bytes': self.torch.cuda.max_memory_allocated(self.device),
                'reserved_bytes': self.torch.cuda.max_memory_reserved(self.device)}
