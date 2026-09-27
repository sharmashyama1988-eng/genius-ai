"""Qwen2.5-0.5B-Instruct Inference Engine with real-time streaming and device adaptation."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of full text generation."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    duration_sec: float
    tokens_per_sec: float


def resolve_model_id(model_id: str) -> str:
    clean = model_id.strip()
    if clean.lower() in (
        "qwen2.5-0.5b-chat",
        "qwen/qwen2.5-0.5b-chat",
        "qwen2.5-0.5b-instruct",
        "qwen/qwen2.5-0.5b-instruct",
    ):
        return "Qwen/Qwen2.5-0.5B-Instruct"
    return clean


class QwenEngine:
    """High-efficiency inference engine for Qwen2.5-0.5B-Chat / Instruct."""

    def __init__(
        self,
        model_id: str = "Qwen2.5-0.5B-Chat",
        device: Optional[str] = None,
        load_on_init: bool = False,
    ) -> None:
        self.display_name = model_id
        self.model_id = resolve_model_id(model_id)
        self._device = device
        self.tokenizer = None
        self.model = None
        self._is_loaded = False
        self._lock = threading.Lock()

        if load_on_init:
            self.load()

    @property
    def device(self) -> str:
        if self._device is not None:
            return self._device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        """Load model and tokenizer into memory."""
        if self._is_loaded:
            return

        with self._lock:
            if self._is_loaded:
                return

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            target_device = self.device
            logger.info(f"Loading {self.model_id} on target device: {target_device}...")
            start_time = time.time()

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )

            if target_device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                ).to("cpu")

            self.model.eval()
            self._is_loaded = True
            load_time = time.time() - start_time
            logger.info(f"Successfully loaded {self.model_id} on {target_device} in {load_time:.2f}s")

    def _ensure_loaded(self) -> None:
        if not self._is_loaded:
            self.load()

    def format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Applies chat template according to Qwen format."""
        self._ensure_loaded()
        assert self.tokenizer is not None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 1536,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> AsyncIterator[str]:
        """Asynchronously stream generated tokens using TextIteratorStreamer."""
        self._ensure_loaded()
        import torch
        from transformers import TextIteratorStreamer

        prompt = self.format_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)  # type: ignore

        streamer = TextIteratorStreamer(
            self.tokenizer,  # type: ignore
            skip_prompt=True,
            skip_special_tokens=False,
        )

        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0.0,
            "pad_token_id": self.tokenizer.eos_token_id,  # type: ignore
        }

        # Run model generation in background thread
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)  # type: ignore
        thread.start()

        loop = asyncio.get_running_loop()

        # Stop tokens in Qwen2.5
        stop_tokens = {"<|im_end|>", "<|endoftext|>"}

        def get_next():
            try:
                return next(streamer)
            except StopIteration:
                return None

        while True:
            token = await loop.run_in_executor(None, get_next)
            if token is None:
                break
            if any(stop_str in token for stop_str in stop_tokens):
                cleaned = token
                for stop_str in stop_tokens:
                    cleaned = cleaned.replace(stop_str, "")
                if cleaned:
                    yield cleaned
                break
            yield token

        thread.join()

    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 1536,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> GenerationResult:
        """Generate response synchronously wrapped in async."""
        start_time = time.time()
        tokens: List[str] = []
        async for token in self.stream_generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        ):
            tokens.append(token)

        duration = max(time.time() - start_time, 0.001)
        full_text = "".join(tokens).strip()

        assert self.tokenizer is not None
        prompt_text = self.format_prompt(messages)
        prompt_tokens = len(self.tokenizer.encode(prompt_text))
        completion_tokens = len(self.tokenizer.encode(full_text))

        return GenerationResult(
            text=full_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_sec=round(duration, 3),
            tokens_per_sec=round(completion_tokens / duration, 2),
        )
