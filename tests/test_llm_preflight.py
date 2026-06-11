"""Fail-fast behavior: preflight, stub fallback transparency, cost notes, SIGINT (issue #4)."""
from __future__ import annotations

import signal

import pytest

from dmn import llm as llm_mod
from dmn.llm import StubLLM, estimate_cost_note, get_llm, preflight
from dmn.loop import install_graceful_sigint


class _FailingLLM:
    def __init__(self, name: str, error: str, model: str = "some-model"):
        self.name = name
        self.model = model
        self._error = error

    def complete(self, system, user, max_tokens=1024):
        raise RuntimeError(self._error)


class _WorkingLLM:
    name = "anthropic"
    model = "claude-sonnet-4-6"

    def complete(self, system, user, max_tokens=1024):
        return llm_mod.LLMResponse(text="ok", model=self.model)


def test_preflight_passes_stub_and_working_provider():
    assert preflight(StubLLM()) is None
    assert preflight(_WorkingLLM()) is None


def test_preflight_classifies_bad_key():
    msg = preflight(_FailingLLM("anthropic", "Error code: 401 - invalid x-api-key"))
    assert "ANTHROPIC_API_KEY" in msg and "rejected" in msg


def test_preflight_classifies_missing_model():
    msg = preflight(_FailingLLM("anthropic", "Error code: 404 - model not_found"))
    assert "DMN_LLM_MODEL" in msg and "some-model" in msg


def test_preflight_local_server_hint():
    msg = preflight(_FailingLLM("ollama", "Connection refused"))
    assert "ollama" in msg and "serve" in msg.lower()


def test_preflight_hosted_network_error_is_not_blamed_on_key():
    msg = preflight(_FailingLLM("anthropic", "Connection timed out via proxy authentication"))
    assert "network" in msg.lower()
    assert "rejected" not in msg


def test_preflight_times_out_on_stalled_server():
    import time

    class _HangingLLM:
        name = "anthropic"
        model = "claude-sonnet-4-6"

        def complete(self, system, user, max_tokens=1024):
            time.sleep(5)

    msg = preflight(_HangingLLM(), timeout_s=0.2)
    assert msg is not None and "no response" in msg


def test_get_llm_records_why_it_fell_back_to_stub(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DMN_LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    llm = get_llm()
    assert llm.name == "stub"
    assert "ANTHROPIC_API_KEY" in llm.fallback_reason


def test_dry_run_stub_is_deliberate_not_a_fallback():
    assert get_llm(dry_run=True).fallback_reason is None


def test_anthropic_default_model_is_current(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    monkeypatch.delenv("DMN_LLM_MODEL", raising=False)
    llm = get_llm("anthropic")
    assert llm.model == "claude-sonnet-4-6"


def test_cost_note_by_provider():
    assert "demo" in estimate_cost_note(StubLLM())
    assert "free" in estimate_cost_note(_FailingLLM("ollama", "x"))
    note = estimate_cost_note(_WorkingLLM(), iterations=12)
    assert "$" in note and "36 LLM calls" in note


class _Console:
    def __init__(self):
        self.lines = []

    def print(self, msg):
        self.lines.append(str(msg))


def test_announce_blocks_broken_provider():
    console = _Console()
    with pytest.raises(SystemExit):
        llm_mod.announce_and_preflight(
            _FailingLLM("anthropic", "401 unauthorized"), console, dry_run=False
        )
    assert any("rejected" in l for l in console.lines)


def test_announce_warns_loudly_on_stub_without_blocking_automation():
    console = _Console()
    llm_mod.announce_and_preflight(
        StubLLM("no keys found"), console, dry_run=False
    )  # non-tty stdin under pytest: warn, don't prompt
    joined = "\n".join(console.lines)
    assert "demo" in joined and "no keys found" in joined


def test_announce_is_quiet_for_dry_run():
    console = _Console()
    llm_mod.announce_and_preflight(StubLLM(), console, dry_run=True)
    assert len(console.lines) == 1  # banner only, no warning


def test_graceful_sigint_two_stage():
    old = signal.getsignal(signal.SIGINT)
    try:
        state = install_graceful_sigint()
        handler = signal.getsignal(signal.SIGINT)
        handler(signal.SIGINT, None)
        assert state["stop"] is True
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGINT, None)
    finally:
        signal.signal(signal.SIGINT, old)
