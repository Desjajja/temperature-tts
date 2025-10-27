
import random
import re
from typing import Dict, Any, List, Optional

BOXED_PATTERN = re.compile(r"\\boxed\{(.+?)\}")

class BaseSampler:
    def __init__(self, model_name: str, max_tokens: int = 256):
        self.model_name = model_name
        self.max_tokens = max_tokens

    def generate_one(self, prompt: str, temperature: float) -> str:
        raise NotImplementedError

    def generate_many(self, prompts: List[str], temperature: float) -> List[str]:
        """Default batching fallback: sequential generation."""
        return [self.generate_one(prompt, temperature=temperature) for prompt in prompts]

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
    def __init__(self, model_name: str, max_tokens: int = 256, device: Optional[str] = None, *, dtype: Optional[str] = None, device_map: Optional[str] = None):
        super().__init__(model_name, max_tokens)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch  # noqa
        except Exception as e:
            raise RuntimeError("Transformers not available. Install with `pip install transformers accelerate`.") from e

        import torch  # noqa
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # Prefer left padding for causal LM batch generation; ignore if tokenizer doesn't expose it
        try:
            self.tokenizer.padding_side = "left"
        except Exception:
            pass

        requested_device_map = device_map if device_map not in (None, "auto") else "auto"

        resolved_dtype: Optional["torch.dtype"]
        if dtype is None:
            resolved_dtype = torch.float16  # type: ignore[attr-defined]
        else:
            dtype_lower = dtype.lower()
            if dtype_lower in {"float16", "fp16", "half"}:
                resolved_dtype = torch.float16
            elif dtype_lower in {"bfloat16", "bf16"}:
                resolved_dtype = torch.bfloat16
            elif dtype_lower in {"float32", "fp32"}:
                resolved_dtype = torch.float32
            elif dtype_lower == "auto":
                resolved_dtype = None
            else:
                raise ValueError(f"Unsupported dtype requested: {dtype}")

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": requested_device_map,
        }
        try:
            if resolved_dtype is not None:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=resolved_dtype,  # type: ignore[arg-type]
                    **model_kwargs,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs,
                )
        except TypeError:
            if resolved_dtype is not None:
                model_kwargs_with_torch = dict(model_kwargs)
                model_kwargs_with_torch["torch_dtype"] = resolved_dtype
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs_with_torch,
                )
            else:
                raise
        # Ensure eval mode for inference
        try:
            self.model.eval()
        except Exception:
            pass

    def _generate(self, tokenized_inputs: "torch.Tensor", attention_mask: "torch.Tensor", temperature: float) -> "torch.Tensor":
        import torch
        with torch.inference_mode():
            return self.model.generate(
                input_ids=tokenized_inputs,
                attention_mask=attention_mask,
                max_new_tokens=self.max_tokens,
                do_sample=(temperature > 0.0),
                temperature=max(1e-6, float(temperature)),
                top_p=1.0,
            )

    def generate_one(self, prompt: str, temperature: float) -> str:
        return self.generate_many([prompt], temperature=temperature)[0]

    def generate_many(self, prompts: List[str], temperature: float) -> List[str]:
        if not prompts:
            return []
        # Use chat template if available (e.g., Qwen/Llama chat models). Fallback to raw prompts.
        prompt_texts: List[str]
        apply_ct = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply_ct):
            try:
                prompt_texts = [
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": p}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for p in prompts
                ]
            except Exception:
                prompt_texts = prompts
        else:
            prompt_texts = prompts

        tk = self.tokenizer(prompt_texts, return_tensors="pt", padding=True)
        tk = {k: v.to(self.model.device) for k, v in tk.items()}
        out = self._generate(tk["input_ids"], tk["attention_mask"], temperature=temperature)
        if out.shape[0] != len(prompts):
            raise RuntimeError(f"Expected {len(prompts)} generations but received {out.shape[0]}")
        return [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in out]

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

def get_sampler(model_name: str, max_tokens: int = 256, *, base_url: Optional[str] = None, api_key: Optional[str] = None, extra_headers: Optional[Dict[str, str]] = None, dtype: Optional[str] = None, device_map: Optional[str] = None):
    """
    Return an appropriate sampler implementation.

    Priority:
    - "dummy" model always uses DummySampler (offline; no network calls).
    - If an OpenAI-compatible endpoint info is provided, use OpenAISampler.
    - Otherwise, fall back to TransformersSampler.
    """
    # Never route the dummy model through network-backed samplers
    if model_name.lower() == "dummy":
        return DummySampler(model_name, max_tokens=max_tokens)

    # If an OpenAI-compatible base_url or api_key is provided, use the OpenAI sampler
    if (base_url is not None) or (api_key is not None):
        return OpenAISampler(
            model_name,
            max_tokens=max_tokens,
            base_url=base_url,
            api_key=api_key,
            extra_headers=extra_headers,
        )

    # Default to local Transformers
    return TransformersSampler(model_name, max_tokens=max_tokens, dtype=dtype, device_map=device_map)
