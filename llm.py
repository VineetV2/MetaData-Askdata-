"""
llm.py
------
LLM backend abstraction for OpenAI and OpenAI-compatible servers.

Factory:
  make_backend("openai", model_id="gpt-5.2")

Backends are cached so repeated calls in the same process do not reload weights.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple


class LLMBackend:
    def generate(self, messages: List[Dict[str, str]],
                 max_new_tokens: int = 128, **gen_kwargs) -> str:
        raise NotImplementedError

    def generate_with_meta(self, messages: List[Dict[str, str]],
                           max_new_tokens: int = 128, **gen_kwargs) -> Dict[str, Any]:
        txt = self.generate(messages, max_new_tokens=max_new_tokens, **gen_kwargs)
        return {"text": txt, "raw": txt, "thoughts": ""}


class OpenAIBackend(LLMBackend):
    def __init__(
        self,
        model_id: str = "gpt-5.2",
        api_key: Optional[str] = None,
        timeout_seconds: int = 600,
    ):
        self.model_id = model_id
        self.kind = "openai"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout_seconds = timeout_seconds
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, timeout=timeout_seconds)
            self._legacy_openai = None
        except ImportError:
            import openai
            openai.api_key = self.api_key
            self._client = None
            self._legacy_openai = openai

    def generate(self, messages: List[Dict[str, str]],
                 max_new_tokens: int = 128, **gen_kwargs) -> str:
        token_param = (
            "max_completion_tokens" if self.model_id in ["gpt-5-mini", "gpt-5.2"]
            else "max_tokens"
        )
        call_kwargs = {
            "model": self.model_id,
            "messages": messages,
            "temperature": gen_kwargs.get("temperature", 0),
            token_param: max_new_tokens,
        }
        if "seed" in gen_kwargs and gen_kwargs["seed"] is not None:
            call_kwargs["seed"] = gen_kwargs["seed"]
        if self._client is not None:
            response = self._client.chat.completions.create(
                **call_kwargs,
                timeout=self.timeout_seconds,
            )
            return (response.choices[0].message.content or "").strip()
        response = self._legacy_openai.ChatCompletion.create(
            **call_kwargs,
            request_timeout=self.timeout_seconds,
        )
        return (response["choices"][0]["message"]["content"] or "").strip()

    def generate_with_meta(self, messages: List[Dict[str, str]],
                           max_new_tokens: int = 128, **gen_kwargs) -> Dict[str, Any]:
        text = self.generate(messages, max_new_tokens=max_new_tokens, **gen_kwargs)
        return {"text": text, "raw": text, "thoughts": ""}


class OpenAICompatibleBackend(LLMBackend):
    """OpenAI-compatible chat backend, suitable for vLLM's /v1 server."""

    def __init__(
        self,
        model_id: str,
        api_base: str = "http://127.0.0.1:8000/v1",
        api_key: str = "EMPTY",
        timeout_seconds: int = 600,
    ):
        self.model_id = model_id
        self.kind = "openai_compatible"
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key or "EMPTY"
        self.timeout_seconds = timeout_seconds
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=timeout_seconds,
            )
            self._legacy_openai = None
        except ImportError:
            import openai
            openai.api_base = self.api_base
            openai.api_key = self.api_key
            self._client = None
            self._legacy_openai = openai

    def generate(self, messages: List[Dict[str, str]],
                 max_new_tokens: int = 128, **gen_kwargs) -> str:
        call_kwargs = {
            "model": self.model_id,
            "messages": messages,
            "temperature": gen_kwargs.get("temperature", 0),
            "max_tokens": max_new_tokens,
        }
        if "seed" in gen_kwargs and gen_kwargs["seed"] is not None:
            call_kwargs["seed"] = gen_kwargs["seed"]
        if self._client is not None:
            response = self._client.chat.completions.create(
                **call_kwargs,
                timeout=self.timeout_seconds,
            )
            return (response.choices[0].message.content or "").strip()
        previous_api_base = getattr(self._legacy_openai, "api_base", None)
        previous_api_key = getattr(self._legacy_openai, "api_key", None)
        self._legacy_openai.api_base = self.api_base
        self._legacy_openai.api_key = self.api_key
        try:
            response = self._legacy_openai.ChatCompletion.create(
                **call_kwargs,
                request_timeout=self.timeout_seconds,
            )
            return (response["choices"][0]["message"]["content"] or "").strip()
        finally:
            if previous_api_base is not None:
                self._legacy_openai.api_base = previous_api_base
            if previous_api_key is not None:
                self._legacy_openai.api_key = previous_api_key


_BACKEND_CACHE: Dict[Tuple, LLMBackend] = {}


def clear_backend_cache() -> None:
    """Clear cached backends (use when switching models)."""
    _BACKEND_CACHE.clear()


def make_backend(kind: str, model_id: Optional[str] = None,
                 cache: bool = True, **kwargs) -> LLMBackend:
    """Create (or retrieve cached) OpenAI LLM backend."""
    k = (kind or "").lower().strip()

    if k in ("openai", "gpt"):
        mid = model_id or "gpt-5.2"
        key = ("openai", mid)
        if cache and key in _BACKEND_CACHE:
            return _BACKEND_CACHE[key]
        backend: LLMBackend = OpenAIBackend(
            mid,
            timeout_seconds=int(kwargs.get("timeout_seconds", 600)),
        )
        if cache:
            _BACKEND_CACHE[key] = backend
        return backend

    if k in ("vllm", "openai-compatible", "openai_compatible"):
        mid = model_id or kwargs.get("model") or ""
        if not mid:
            raise ValueError("model_id is required for vLLM/OpenAI-compatible backends")
        api_base = kwargs.get("api_base", "http://127.0.0.1:8000/v1")
        api_key = kwargs.get("api_key", "EMPTY")
        timeout_seconds = int(kwargs.get("timeout_seconds", 600))
        key = ("openai_compatible", mid, api_base, api_key, timeout_seconds)
        if cache and key in _BACKEND_CACHE:
            return _BACKEND_CACHE[key]
        backend = OpenAICompatibleBackend(
            mid,
            api_base=api_base,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        if cache:
            _BACKEND_CACHE[key] = backend
        return backend

    raise ValueError(f"Unknown backend kind: {kind!r}. Use 'openai' or 'vllm'.")
