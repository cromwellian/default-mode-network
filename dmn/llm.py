"""LLM providers: anthropic, openai, ollama, lmstudio, and a stub for dry-run / no-key envs.

Local providers (`ollama`, `lmstudio`, `vllm`) reuse the OpenAI SDK pointed at a custom `base_url`.
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
    "vllm": {
        "base_url": "http://localhost:8000/v1",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "hint": "Is the vLLM OpenAI server running? Set DMN_LLM_BASE_URL to its /v1 endpoint.",
    },
}


class StubLLM:
    """A no-API echo LLM for dry-run mode. Templates a useful-ish response from the prompt."""

    name = "stub"
    model = "stub"

    def __init__(self, fallback_reason: Optional[str] = None) -> None:
        # Set when get_llm() landed here because a real provider failed, so
        # entry points can tell the user *why* they're in demo mode.
        self.fallback_reason = fallback_reason

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

    def __init__(self, model: Optional[str] = None) -> None:
        try:
            import anthropic  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("anthropic not installed; `uv add anthropic`") from e
        self.client = anthropic.Anthropic()
        self.model = (
            model
            or os.environ.get("DMN_LLM_MODEL")
            or "claude-sonnet-4-6"
        )

    # Above this many output tokens the SDK refuses a non-streaming call ("Streaming is
    # required for operations that may take longer than 10 minutes"), so large code/HTML
    # activities must stream. Small calls (synthesis, seed brainstorming) stay non-streaming.
    _STREAM_THRESHOLD = 4096

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> LLMResponse:
        """Send a single-turn message to Claude and return the text body."""
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if max_tokens > self._STREAM_THRESHOLD:
            with self.client.messages.stream(**kwargs) as stream:
                msg = stream.get_final_message()
        else:
            msg = self.client.messages.create(**kwargs)
        text = "".join(getattr(b, "text", "") for b in msg.content)
        return LLMResponse(text=text, model=self.model)


class OpenAILLM:
    """OpenAI Chat Completions provider. Reads OPENAI_API_KEY from env."""

    name = "openai"

    def __init__(self, model: Optional[str] = None) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("openai not installed; `uv add openai`") from e
        self.client = OpenAI()
        self.model = model or os.environ.get("DMN_LLM_MODEL") or "gpt-4o-mini"

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
    if provider in ("ollama", "lmstudio", "vllm"):
        try:
            return LocalOpenAICompatibleLLM(provider)
        except Exception as e:
            return StubLLM(f"DMN_LLM_PROVIDER={provider}, but it failed to initialize: {e}")
    if provider == "anthropic":
        try:
            return AnthropicLLM()
        except Exception as e:
            return StubLLM(f"DMN_LLM_PROVIDER=anthropic, but it failed to initialize: {e}")
    if provider == "openai":
        try:
            return OpenAILLM()
        except Exception as e:
            return StubLLM(f"DMN_LLM_PROVIDER=openai, but it failed to initialize: {e}")
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
    return StubLLM(
        "no ANTHROPIC_API_KEY or OPENAI_API_KEY found and no DMN_LLM_PROVIDER set"
    )


# Rough per-iteration cost band for hosted sonnet/4o-class models: ~3 LLM calls
# of a few thousand tokens each. Honest order-of-magnitude, not a quote.
_COST_PER_ITER_LOW = 0.02
_COST_PER_ITER_HIGH = 0.07
_CALLS_PER_ITER = 3


def estimate_cost_note(llm, iterations: int | None = None, minutes: float | None = None) -> str:
    """One short human line about what a run will cost with this provider."""
    if llm.name == "stub":
        return "no API calls (demo mode)"
    if llm.name in _LOCAL_DEFAULTS:
        return "local model — free, but slower than a hosted API"
    iters = iterations if iterations is not None else max(1, round((minutes or 12) * 1.0))
    lo, hi = iters * _COST_PER_ITER_LOW, iters * _COST_PER_ITER_HIGH
    return f"≈{iters * _CALLS_PER_ITER} LLM calls, roughly ${lo:.2f}–${hi:.2f}"


def preflight(llm) -> str | None:
    """Make one cheap completion to fail fast on bad keys/models/servers.

    Returns a friendly, actionable error string, or None when the provider works.
    """
    if llm.name == "stub":
        return None
    try:
        llm.complete("Reply with the single word: ok", "ok?", max_tokens=1)
        return None
    except Exception as e:
        s = str(e)
        low = s.lower()
        if any(t in low for t in ("401", "authentication", "unauthorized", "api key", "api_key")):
            var = "OPENAI_API_KEY" if llm.name == "openai" else "ANTHROPIC_API_KEY"
            return (
                f"{llm.name}: the API rejected your key. Check {var} in .env "
                f"(detail: {s[:120]})"
            )
        if any(t in low for t in ("404", "not_found", "not found", "does not exist")):
            return (
                f"{llm.name}: model '{llm.model}' is unavailable. Set DMN_LLM_MODEL to a "
                f"current model or unset it for the default (detail: {s[:120]})"
            )
        if llm.name in _LOCAL_DEFAULTS:
            return (
                f"{llm.name}: can't reach the local server. "
                f"{_LOCAL_DEFAULTS[llm.name]['hint']} (detail: {s[:80]})"
            )
        return f"{llm.name}: provider check failed: {s[:160]}"


def announce_and_preflight(
    llm,
    console,
    dry_run: bool = False,
    iterations: int | None = None,
    minutes: float | None = None,
) -> None:
    """Print the provider banner, then fail fast (or warn loudly) before any real run.

    Raises SystemExit(1) when the configured provider doesn't work, or when the user
    declines to continue an interactive non-dry run that fell back to the stub LLM.
    """
    console.print(
        f"LLM provider: [bold]{llm.name}[/] · model [bold]{llm.model}[/] · "
        f"{estimate_cost_note(llm, iterations=iterations, minutes=minutes)}"
    )
    if dry_run:
        return
    if llm.name == "stub":
        console.print(
            "[red]⚠ No working LLM — this run will produce templated demo briefs, "
            "not real research.[/]"
        )
        reason = getattr(llm, "fallback_reason", None)
        if reason:
            console.print(f"[red]  why: {reason}[/]")
        console.print(
            "[red]  Fix: put ANTHROPIC_API_KEY in .env, or set DMN_LLM_PROVIDER=ollama "
            "for a local model. Use --dry-run if demo mode is what you want.[/]"
        )
        if sys.stdin.isatty():
            ans = input("Continue in demo mode anyway? [y/N] ").strip().lower()
            if ans not in ("y", "yes"):
                raise SystemExit(1)
        return
    err = preflight(llm)
    if err:
        console.print(f"[red]{err}[/]")
        raise SystemExit(1)
