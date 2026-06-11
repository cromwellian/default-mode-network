"""Hash-fallback + embedding-compat behavior when sentence-transformers is absent (issue #3)."""
from __future__ import annotations

import numpy as np
import pytest

from dmn import embeddings, portability, store


@pytest.fixture(autouse=True)
def st_unavailable(monkeypatch):
    monkeypatch.setattr(embeddings, "_get_st_model", lambda: None)
    monkeypatch.setattr(embeddings, "_warned_fallback", False)
    monkeypatch.delenv("DMN_EMBEDDINGS", raising=False)


def test_embed_falls_back_to_hash_with_one_notice(capsys):
    out = embeddings.embed(["fermentation", "jazz piano"])
    assert out.shape == (2, embeddings.DEFAULT_DIM)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    embeddings.embed(["again"])
    err = capsys.readouterr().err
    assert err.count("hash fallback") == 1
    assert "--extra embeddings" in err


def test_explicit_hash_backend_is_silent(monkeypatch, capsys):
    monkeypatch.setenv("DMN_EMBEDDINGS", "hash")
    embeddings.embed(["fermentation"])
    assert capsys.readouterr().err == ""


def test_effective_backend_resolves_missing_st_to_hash():
    assert embeddings.effective_backend() == "hash"


def test_profile_fingerprint_reflects_hash_fallback():
    assert portability._embedding_model_name() == "dmn/hash-fallback"


def test_fingerprint_uses_st_name_when_st_available(monkeypatch):
    monkeypatch.setattr(embeddings, "_get_st_model", lambda: object())
    assert portability._embedding_model_name() == "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture()
def conn(tmp_path):
    c = store.connect(tmp_path / "dmn.sqlite")
    yield c
    c.close()


def test_profile_mismatch_none_when_backends_match(conn):
    portability.stamp_profile_embedding(conn)
    assert portability.profile_embedding_mismatch(conn) is None


def test_profile_mismatch_none_for_unstamped_legacy_profile(conn):
    assert portability.profile_embedding_mismatch(conn) is None


def test_profile_mismatch_detected_with_install_hint(conn, monkeypatch):
    monkeypatch.setattr(embeddings, "_get_st_model", lambda: object())
    portability.stamp_profile_embedding(conn)  # stamped as sentence-transformers
    monkeypatch.setattr(embeddings, "_get_st_model", lambda: None)  # ST later pruned
    msg = portability.profile_embedding_mismatch(conn)
    assert msg is not None
    assert "sentence-transformers/all-MiniLM-L6-v2" in msg
    assert "--extra embeddings" in msg


def test_profile_mismatch_detected_on_backend_switch(conn, monkeypatch):
    portability.stamp_profile_embedding(conn)  # stamped as hash-fallback
    monkeypatch.setenv("DMN_EMBEDDINGS", "openai")
    msg = portability.profile_embedding_mismatch(conn)
    assert msg is not None
    assert "prepare.py" in msg
