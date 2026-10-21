import pytest

pytest.importorskip("pydantic_settings")

from mmsearch.config import Settings  # noqa: E402


def test_defaults_are_consistent():
    s = Settings(_env_file=None)
    assert s.k_chunks_for(s.rerank_candidates) >= s.rerank_candidates
    assert s.ef_for(200) == 200


def test_rejects_bad_overlap():
    with pytest.raises(ValueError):
        Settings(_env_file=None, chunk_tokens=64, chunk_overlap=40)


def test_env_override(monkeypatch):
    monkeypatch.setenv("MMS_RERANK_CANDIDATES", "17")
    assert Settings(_env_file=None).rerank_candidates == 17


def test_image_settings():
    s = Settings(_env_file=None)
    assert s.clip_dim == 512 and s.enable_images
    with pytest.raises(ValueError):
        Settings(_env_file=None, caption_weight=1.5)
