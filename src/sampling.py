
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

class VLLMSampler(BaseSampler):
    """
    Sampler using vLLM engine (Python API).
    Allows fine-grained control via SamplingParams and engine init args.
    """
    def __init__(
        self,
        model_name: str,
        max_tokens: int = 256,
        *,
        dtype: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 0,
        max_model_len: Optional[int] = None,
        # default sampling params
        vllm_top_p: Optional[float] = None,
        vllm_top_k: Optional[int] = None,
        vllm_repetition_penalty: Optional[float] = None,
        vllm_presence_penalty: Optional[float] = None,
        vllm_frequency_penalty: Optional[float] = None,
    ):
        super().__init__(model_name, max_tokens)
        try:
            from vllm import LLM  # type: ignore
        except Exception as e:
            raise RuntimeError("vLLM not available. Install with `pip install vllm`.") from e

        # Initialize engine
        engine_kwargs: Dict[str, Any] = {
            "model": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "swap_space": swap_space,
            "trust_remote_code": True,
        }
        if dtype is not None:
            engine_kwargs["dtype"] = dtype
        if max_model_len is not None and max_model_len > 0:
            engine_kwargs["max_model_len"] = max_model_len

        self._llm = LLM(**engine_kwargs)

        # Store default sampling params (can be None to use vLLM defaults)
        self._default_sampling: Dict[str, Any] = {
            "top_p": vllm_top_p,
            "top_k": vllm_top_k,
            "repetition_penalty": vllm_repetition_penalty,
            "presence_penalty": vllm_presence_penalty,
            "frequency_penalty": vllm_frequency_penalty,
        }

        # Optional chat template support via HF tokenizer (best-effort)
        self._apply_chat_template = None
        try:
            from transformers import AutoTokenizer  # type: ignore
            _tk = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if hasattr(_tk, "apply_chat_template"):
                def _tmpl(msgs: List[Dict[str, str]]) -> str:
                    return _tk.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True
                    )
                self._apply_chat_template = _tmpl
        except Exception:
            self._apply_chat_template = None

    def _to_prompt_texts(self, prompts: List[str]) -> List[str]:
        if not prompts:
            return []
        if self._apply_chat_template is None:
            return prompts
        out: List[str] = []
        for p in prompts:
            try:
                out.append(self._apply_chat_template([{ "role": "user", "content": p }]))
            except Exception:
                out.append(p)
        return out

    def generate_one(self, prompt: str, temperature: float) -> str:
        return self.generate_many([prompt], temperature=temperature)[0]

    def generate_many(self, prompts: List[str], temperature: float) -> List[str]:
        if not prompts:
            return []
        from vllm import SamplingParams  # type: ignore

        prompt_texts = self._to_prompt_texts(prompts)

        sp_kwargs: Dict[str, Any] = {
            "temperature": max(1e-6, float(temperature)),
            "max_tokens": self.max_tokens,
        }
        # Merge defaults if provided
        for k, v in self._default_sampling.items():
            if v is not None:
                sp_kwargs[k] = v
        sampling_params = SamplingParams(**sp_kwargs)

        outputs = self._llm.generate(prompt_texts, sampling_params)
        # vLLM returns one RequestOutput per prompt, each with .outputs (list), select first
        texts: List[str] = []
        for out in outputs:
            try:
                if out.outputs:
                    texts.append(out.outputs[0].text)
                else:
                    texts.append("")
            except Exception:
                texts.append("")
        return texts


def get_sampler(
    model_name: str,
    max_tokens: int = 256,
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    dtype: Optional[str] = None,
    device_map: Optional[str] = None,
    backend: Optional[str] = None,
    # vLLM engine args
    vllm_tensor_parallel_size: Optional[int] = None,
    vllm_gpu_memory_utilization: Optional[float] = None,
    vllm_swap_space: Optional[int] = None,
    vllm_max_model_len: Optional[int] = None,
    # vLLM sampling defaults
    vllm_top_p: Optional[float] = None,
    vllm_top_k: Optional[int] = None,
    vllm_repetition_penalty: Optional[float] = None,
    vllm_presence_penalty: Optional[float] = None,
    vllm_frequency_penalty: Optional[float] = None,
):
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
    if (base_url is not None) or (api_key is not None) or (backend == "openai"):
        return OpenAISampler(
            model_name,
            max_tokens=max_tokens,
            base_url=base_url,
            api_key=api_key,
            extra_headers=extra_headers,
        )

    # vLLM backend explicitly requested
    if backend == "vllm":
        return VLLMSampler(
            model_name,
            max_tokens=max_tokens,
            dtype=dtype,
            tensor_parallel_size=vllm_tensor_parallel_size or 1,
            gpu_memory_utilization=vllm_gpu_memory_utilization or 0.9,
            swap_space=vllm_swap_space or 0,
            max_model_len=vllm_max_model_len if (vllm_max_model_len and vllm_max_model_len > 0) else None,
            vllm_top_p=vllm_top_p,
            vllm_top_k=vllm_top_k,
            vllm_repetition_penalty=vllm_repetition_penalty,
            vllm_presence_penalty=vllm_presence_penalty,
            vllm_frequency_penalty=vllm_frequency_penalty,
        )

    # Default or 'transformers' backend
    return TransformersSampler(model_name, max_tokens=max_tokens, dtype=dtype, device_map=device_map)
