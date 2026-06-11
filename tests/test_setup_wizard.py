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


def test_interview_numbers_prompts_and_echoes_captured_pieces():
    from dmn.importers import manual

    fed = iter(["fermentation, jazz piano", "", "x", "", "", ""])
    seen: list[str] = []
    answers = manual.interview(input_fn=lambda q: (seen.append(q), next(fed))[1], output_fn=seen.append)
    texts = [a["text"] for a in answers]
    assert texts == ["fermentation", "jazz piano"]
    joined = "\n".join(seen)
    assert "[1/6]" in joined and "[6/6]" in joined
    assert "noted: fermentation; jazz piano" in joined
    assert "too short" in joined


def test_machine_arch_sees_through_rosetta(monkeypatch):
    from dmn import setup_wizard as sw

    monkeypatch.setattr(sw.platform, "machine", lambda: "x86_64")
    # NB: sys is shared — _total_ram_gb() also reads sys.platform; keep this
    # patch away from tests that call it.
    monkeypatch.setattr(sw.sys, "platform", "darwin")

    class _Out:
        stdout = "1\nApple M2 Max\n"

    monkeypatch.setattr(sw.subprocess, "run", lambda *a, **k: _Out())
    assert sw._machine_arch() == "arm64"

    class _Intel:
        stdout = "0\nIntel(R) Core(TM) i9\n"

    monkeypatch.setattr(sw.subprocess, "run", lambda *a, **k: _Intel())
    assert sw._machine_arch() == "x86_64"


def test_machine_arch_passthrough_off_darwin(monkeypatch):
    from dmn import setup_wizard as sw

    monkeypatch.setattr(sw.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(sw.sys, "platform", "linux")
    assert sw._machine_arch() == "x86_64"


def _scripted_walk(confirms, asks):
    from dmn import setup_wizard as sw

    c = iter(confirms)
    a = iter(asks)
    said = []
    return sw.source_walk(
        ask=lambda *p, **k: next(a),
        confirm=lambda *p, **k: next(c),
        say=said.append,
    )


def test_source_walk_all_skipped_yields_no_args():
    assert _scripted_walk([False, False, False, False, False], [""]) == []


def test_source_walk_interview_plus_browser():
    args = _scripted_walk([True, True, False, False, False], [""])
    assert args == ["--interactive", "--import", "browser"]


def test_source_walk_full_house(tmp_path, monkeypatch):
    from dmn import setup_wizard as sw

    takeout = tmp_path / "Takeout"
    takeout.mkdir()
    tw = tmp_path / "twitter-archive"
    tw.mkdir()
    csv = tmp_path / "liked.csv"
    csv.write_text("title\nsong\n")
    paths = iter([str(takeout), str(tw), str(csv)])
    monkeypatch.setattr(sw, "_ask_path", lambda prompt: next(paths))
    args = sw.source_walk(
        ask=lambda *p, **k: "",
        confirm=lambda *p, **k: True,
        say=lambda *p, **k: None,
    )
    assert "--interactive" in args
    assert ["--twitter-dir", str(tw)] == args[args.index("--twitter-dir"):args.index("--twitter-dir") + 2]
    assert ["--takeout-dir", str(takeout)] == args[args.index("--takeout-dir"):args.index("--takeout-dir") + 2]
    assert ["--readwise-csv", str(csv)] == args[args.index("--readwise-csv"):args.index("--readwise-csv") + 2]
    imports = args[args.index("--import") + 1]
    assert imports == "browser,youtube,gmail,drive,twitter,readwise"
