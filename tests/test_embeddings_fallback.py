"""Hash-fallback behavior when sentence-transformers is not installed (issue #3)."""
from __future__ import annotations

import numpy as np
import pytest

from dmn import embeddings, portability


@pytest.fixture(autouse=True)
def st_unavailable(monkeypatch):
    monkeypatch.setattr(embeddings, "_st_available", lambda: False)
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
    monkeypatch.setattr(embeddings, "_st_available", lambda: True)
    assert portability._embedding_model_name() == "sentence-transformers/all-MiniLM-L6-v2"
