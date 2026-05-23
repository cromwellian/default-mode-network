"""LLM providers: anthropic, openai, ollama, lmstudio, and a stub for dry-run / no-key envs.

Local providers (`ollama`, `lmstudio`) reuse the OpenAI SDK pointed at a custom `base_url`.
Both expose an OpenAI-compatible `/v1/chat/completions` endpoint, so the wire-format code
is identical — they're surfaced as separate provider names purely for UX clarity in env vars
and CLI output.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """A normalized LLM completion result."""

    text: str
    model: str


# Default base URLs and models for each local provider. Override with
# DMN_LLM_BASE_URL / DMN_LLM_MODEL.
_LOCAL_DEFAULTS = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.1:8b",
        "hint": "Is `ollama serve` running? Try: `brew install ollama && ollama pull llama3.1:8b`",
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "model": "local-model",
        "hint": (
            "Is LM Studio's local server running? Open LM Studio → 'Local Server' tab → "
            "load a model → Start Server."
        ),
    },
}


class StubLLM:
    """A no-API echo LLM for dry-run mode. Templates a useful-ish response from the prompt."""

    name = "stub"
    model = "stub"

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> LLMResponse:
        """Return a deterministic, prompt-derived stub response."""
        snippet = (user or "").strip().splitlines()[0] if user else ""
        text = (
            "[stub-llm brief]\n\n"
            f"Seed: {snippet[:160]}\n\n"
            "1. **What surprised me:** the question itself sits at an interesting angle "
            "between two of your taste clusters.\n"
            "2. **Something delightful:** there's a small detail in the gathered findings "
            "worth chasing.\n"
            "3. **Rabbit hole for tomorrow:** keep pulling on this thread; another iteration "
            "with a different tool mix may surface an unexpected connection.\n"
        )
        return LLMResponse(text=text, model=self.model)


class AnthropicLLM:
    """Anthropic Claude provider. Reads ANTHROPIC_API_KEY from env."""

    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-5-20250929") -> None:
        try:
            import anthropic  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("anthropic not installed; `uv add anthropic`") from e
        self.client = anthropic.Anthropic()
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> LLMResponse:
        """Send a single-turn message to Claude and return the text body."""
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(getattr(b, "text", "") for b in msg.content)
        return LLMResponse(text=text, model=self.model)


class OpenAILLM:
    """OpenAI Chat Completions provider. Reads OPENAI_API_KEY from env."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("openai not installed; `uv add openai`") from e
        self.client = OpenAI()
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> LLMResponse:
        """Send a single-turn chat completion and return the assistant text."""
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = resp.choices[0].message.content or ""
        return LLMResponse(text=text, model=self.model)


class LocalOpenAICompatibleLLM:
    """Local LLM via the OpenAI SDK pointed at a custom base_url (Ollama / LM Studio).

    Both Ollama (`/v1`) and LM Studio's local server expose the OpenAI Chat Completions
    schema. We reuse the OpenAI SDK with a dummy api_key. The provider name (`ollama` or
    `lmstudio`) is preserved for log clarity even though the wire format is identical.
    """

    def __init__(
        self,
        provider: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        if provider not in _LOCAL_DEFAULTS:
            raise ValueError(f"unknown local provider: {provider}")
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("openai not installed; `uv add openai`") from e
        defaults = _LOCAL_DEFAULTS[provider]
        self.name = provider
        self.base_url = (
            base_url or os.environ.get("DMN_LLM_BASE_URL") or defaults["base_url"]
        )
        self.model = model or os.environ.get("DMN_LLM_MODEL") or defaults["model"]
        self.client = OpenAI(base_url=self.base_url, api_key="not-needed")
        if not _health_check(self.base_url):
            sys.stderr.write(
                f"[dmn] warning: {provider} unreachable at {self.base_url}. "
                f"{defaults['hint']}\n"
            )

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> LLMResponse:
        """Send a chat completion to the local OpenAI-compatible endpoint."""
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = resp.choices[0].message.content or ""
        return LLMResponse(text=text, model=self.model)


def _health_check(base_url: str, timeout: float = 2.0) -> bool:
    """Quick GET against `<base_url>/models` with a 2s timeout. True on HTTP 200."""
    try:
        import requests  # type: ignore
    except Exception:
        return True  # can't probe; let the actual call surface the error
    try:
        r = requests.get(base_url.rstrip("/") + "/models", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def get_llm(provider: str | None = None, dry_run: bool = False):
    """Return the configured LLM provider; falls back to stub if nothing is reachable."""
    if dry_run:
        return StubLLM()
    provider = (provider or os.environ.get("DMN_LLM_PROVIDER") or "").lower()
    if provider == "stub":
        return StubLLM()
    if provider in ("ollama", "lmstudio"):
        try:
            return LocalOpenAICompatibleLLM(provider)
        except Exception:
            return StubLLM()
    if provider == "anthropic":
        try:
            return AnthropicLLM()
        except Exception:
            return StubLLM()
    if provider == "openai":
        try:
            return OpenAILLM()
        except Exception:
            return StubLLM()
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicLLM()
        except Exception:
            pass
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAILLM()
        except Exception:
            pass
    return StubLLM()
