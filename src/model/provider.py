"""Universal Dual-Core Model Provider: Local Qwen2.5-0.5B + Claude API + LiteLLM + Ollama."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional
import httpx

from .llm_engine import QwenEngine

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for LLM Providers."""
    provider_type: str = "local"  # "local", "claude", "ollama", "openai"
    model_name: str = "Qwen2.5-0.5B-Chat"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    enable_thinking: bool = True
    thinking_budget: int = 2048


class BaseLLMProvider(ABC):
    """Abstract interface for all LLM backends."""

    @abstractmethod
    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 2048,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> AsyncIterator[str]:
        """Streams generated tokens."""
        pass


class LocalQwenProvider(BaseLLMProvider):
    """Runs Qwen2.5-0.5B-Chat natively on local hardware."""

    def __init__(self, model_id: str = "Qwen2.5-0.5B-Chat", device: Optional[str] = None) -> None:
        self.engine = QwenEngine(model_id=model_id, device=device)

    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 2048,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> AsyncIterator[str]:
        async for token in self.engine.stream_generate(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        ):
            yield token


class ClaudeAPIProvider(BaseLLMProvider):
    """Calls Anthropic Claude API with adaptive extended thinking."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "claude-3-7-sonnet-20250219",
        enable_thinking: bool = True,
        thinking_budget: int = 2048,
    ) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.api_url = "https://api.anthropic.com/v1/messages"

    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 4096,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> AsyncIterator[str]:
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not configured. Set environment variable or pass key.")

        # Separate system prompt from conversation
        system_content = ""
        user_messages: List[Dict[str, Any]] = []

        for m in messages:
            if m.get("role") == "system":
                system_content = m.get("content", "")
            else:
                user_messages.append({"role": m.get("role"), "content": m.get("content")})

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": user_messages,
            "max_tokens": max_new_tokens,
            "stream": True,
        }

        if system_content:
            payload["system"] = system_content

        if self.enable_thinking:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }
            payload["temperature"] = 1.0  # Required when thinking is enabled
        else:
            payload["temperature"] = temperature

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", self.api_url, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    err_body = await response.aread()
                    raise RuntimeError(f"Claude API error ({response.status_code}): {err_body.decode('utf-8')}")

                in_thinking_block = False
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            event = json.loads(data_str)
                            event_type = event.get("type")

                            if event_type == "content_block_start":
                                block = event.get("content_block", {})
                                if block.get("type") == "thinking":
                                    in_thinking_block = True
                                    yield "<think>\n"

                            elif event_type == "content_block_delta":
                                delta = event.get("delta", {})
                                if delta.get("type") == "thinking_delta":
                                    yield delta.get("thinking", "")
                                elif delta.get("type") == "text_delta":
                                    if in_thinking_block:
                                        in_thinking_block = False
                                        yield "\n</think>\n"
                                    yield delta.get("text", "")

                            elif event_type == "content_block_stop":
                                if in_thinking_block:
                                    in_thinking_block = False
                                    yield "\n</think>\n"

                        except json.JSONDecodeError:
                            continue


class OllamaProvider(BaseLLMProvider):
    """Connects to local Ollama server instance."""

    def __init__(
        self,
        model_name: str = "qwen2.5:0.5b",
        host: str = "http://localhost:11434",
    ) -> None:
        self.model_name = model_name
        self.host = host.rstrip("/")

    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 2048,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> AsyncIterator[str]:
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_new_tokens,
            },
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    err = await response.aread()
                    raise RuntimeError(f"Ollama error ({response.status_code}): {err.decode('utf-8')}")

                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue


class UniversalModelRouter:
    """Manages active LLM provider and handles auto-failover."""

    def __init__(self, default_provider: str = "local") -> None:
        self.active_provider_name = default_provider
        self.providers: Dict[str, BaseLLMProvider] = {
            "local": LocalQwenProvider(),
        }

        # Auto-configure Claude if key is in environment
        if os.getenv("ANTHROPIC_API_KEY"):
            self.providers["claude"] = ClaudeAPIProvider()

        # Auto-configure Ollama
        self.providers["ollama"] = OllamaProvider()

    def get_provider(self, name: Optional[str] = None) -> BaseLLMProvider:
        target = name or self.active_provider_name
        if target not in self.providers:
            if target == "claude":
                self.providers["claude"] = ClaudeAPIProvider()
            elif target == "ollama":
                self.providers["ollama"] = OllamaProvider()
            else:
                target = "local"
        return self.providers[target]

    def set_provider(self, name: str) -> bool:
        if name in ("local", "claude", "ollama"):
            self.active_provider_name = name
            return True
        return False
