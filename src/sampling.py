
import random
import re
from typing import Dict, Any, List, Tuple, Optional

BOXED_PATTERN = re.compile(r"\\boxed\{(.+?)\}")

class BaseSampler:
    def __init__(self, model_name: str, max_tokens: int = 256):
        self.model_name = model_name
        self.max_tokens = max_tokens

    def generate_one(self, prompt: str, temperature: float) -> str:
        raise NotImplementedError

class DummySampler(BaseSampler):
    """
    A toy sampler that pretends to answer integer questions for the toy dataset.
    """
    def generate_one(self, prompt: str, temperature: float) -> str:
        try:
            if "2 + 3" in prompt:
                base = 5
            elif "12 - 7" in prompt:
                base = 5
            elif "3 * 3" in prompt or "3 \\times 3" in prompt:
                base = 9
            else:
                base = 5
            noise = 0 if temperature == 0.0 else random.choice([0, 0, 0, 1, -1])
            ans = base + noise
            return f"Solution:\\nWe compute.\\n\\n\\\\boxed{{{ans}}}"
        except Exception:
            return "\\boxed{0}"

class TransformersSampler(BaseSampler):
    """
    Optional sampler using HuggingFace Transformers.
    """
    def __init__(self, model_name: str, max_tokens: int = 256, device: Optional[str] = None):
        super().__init__(model_name, max_tokens)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch  # noqa
        except Exception as e:
            raise RuntimeError("Transformers not available. Install with `pip install transformers accelerate`.") from e

        import torch  # noqa
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")

    def generate_one(self, prompt: str, temperature: float) -> str:
        import torch
        tk = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **tk,
            max_new_tokens=self.max_tokens,
            do_sample=(temperature > 0.0),
            temperature=max(1e-6, float(temperature)),
            top_p=1.0,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text

class OpenAISampler(BaseSampler):
    """
    Sampler using an OpenAI-compatible chat completions API (e.g., LiteLLM proxy, vLLM, etc.).
    Streams tokens and concatenates into a final response.
    """
    def __init__(self, model_name: str, max_tokens: int = 256, base_url: Optional[str] = None, api_key: Optional[str] = None, extra_headers: Optional[Dict[str, str]] = None):
        super().__init__(model_name, max_tokens)
        self.base_url = base_url or "http://localhost:4000/chat/v1"
        self.api_key = api_key
        self.extra_headers = extra_headers or {}
        try:
            # Prefer new-style OpenAI client
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("openai package not available. Install with `pip install openai`.") from e
        from openai import OpenAI  # type: ignore
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, default_headers=self.extra_headers or None)

    def generate_one(self, prompt: str, temperature: float) -> str:
        # Use chat completions streaming and concatenate deltas
        content: List[str] = []
        stream = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=max(1e-6, float(temperature)),
            max_tokens=self.max_tokens,
            stream=True,
        )
        # Stream chunks (OpenAI 1.x returns objects with .choices[0].delta.content)
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta  # type: ignore[attr-defined]
                if delta and getattr(delta, "content", None):
                    content.append(delta.content)  # type: ignore[index]
            except Exception:
                # Best-effort: ignore unknown chunk shapes
                pass
        return "".join(content)

def get_sampler(model_name: str, max_tokens: int = 256, *, base_url: Optional[str] = None, api_key: Optional[str] = None, extra_headers: Optional[Dict[str, str]] = None):
    # If an OpenAI-compatible base_url or api_key is provided, use the OpenAI sampler
    if (base_url is not None) or (api_key is not None):
        return OpenAISampler(model_name, max_tokens=max_tokens, base_url=base_url, api_key=api_key, extra_headers=extra_headers)
    if model_name.lower() == "dummy":
        return DummySampler(model_name, max_tokens=max_tokens)
    return TransformersSampler(model_name, max_tokens=max_tokens)
