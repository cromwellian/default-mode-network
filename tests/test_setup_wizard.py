"""Setup-wizard pure logic: hardware-aware model picks and .env editing (issue #5)."""
from __future__ import annotations

from dmn.setup_wizard import recommend_local_model, update_env_file


def test_apple_silicon_16gb_gets_8b():
    model, note = recommend_local_model("arm64", 16)
    assert model == "llama3.1:8b"


def test_apple_silicon_8gb_gets_3b():
    model, note = recommend_local_model("arm64", 8)
    assert model == "llama3.2:3b"


def test_intel_mac_warned_off_8b():
    model, note = recommend_local_model("x86_64", 32)
    assert model == "llama3.2:3b"
    assert "hosted" in note.lower() or "api" in note.lower()


def test_small_machines_steered_to_hosted_or_demo():
    assert recommend_local_model("x86_64", 8)[0] is None
    assert recommend_local_model("arm64", 4)[0] is None


def test_unknown_ram_is_not_treated_as_tiny():
    model, note = recommend_local_model("arm64", None)
    assert model == "llama3.2:3b"
    assert "detect" in note.lower()
    assert recommend_local_model("x86_64", None)[0] is None


def test_env_file_created_fresh(tmp_path):
    p = tmp_path / ".env"
    update_env_file({"ANTHROPIC_API_KEY": "sk-test"}, path=p)
    assert p.read_text() == "ANTHROPIC_API_KEY=sk-test\n"


def test_env_file_preserves_unrelated_lines_and_comments(tmp_path):
    p = tmp_path / ".env"
    p.write_text("# my notes\nHF_TOKEN=hf_abc\nANTHROPIC_API_KEY=old\n")
    update_env_file({"ANTHROPIC_API_KEY": "new", "DMN_LLM_PROVIDER": "ollama"}, path=p)
    text = p.read_text()
    assert "# my notes" in text
    assert "HF_TOKEN=hf_abc" in text
    assert "ANTHROPIC_API_KEY=new" in text
    assert "ANTHROPIC_API_KEY=old" not in text
    assert text.count("ANTHROPIC_API_KEY=") == 1
    assert "DMN_LLM_PROVIDER=ollama" in text


def test_env_file_none_removes_stale_provider(tmp_path):
    p = tmp_path / ".env"
    p.write_text("DMN_LLM_PROVIDER=ollama\nDMN_LLM_MODEL=llama3.1:8b\nHF_TOKEN=x\n")
    update_env_file({"ANTHROPIC_API_KEY": "sk-new", "DMN_LLM_PROVIDER": None, "DMN_LLM_MODEL": None}, path=p)
    text = p.read_text()
    assert "DMN_LLM_PROVIDER" not in text
    assert "DMN_LLM_MODEL" not in text
    assert "HF_TOKEN=x" in text
    assert "ANTHROPIC_API_KEY=sk-new" in text


def test_env_file_is_private(tmp_path):
    p = tmp_path / ".env"
    update_env_file({"ANTHROPIC_API_KEY": "sk-test"}, path=p)
    assert (p.stat().st_mode & 0o777) == 0o600
