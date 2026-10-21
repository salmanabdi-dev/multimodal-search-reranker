import pytest

from mmsearch.ingestion.chunker import TokenChunker
from tests.unit.fakes import WhitespaceTokenizer


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_short_text_is_one_chunk():
    chunks = TokenChunker(WhitespaceTokenizer(), max_tokens=50, overlap=5).chunk(words(10))
    assert len(chunks) == 1
    assert chunks[0].content == words(10)
    assert chunks[0].token_count == 12  # 10 words + 2 specials


def test_empty_text():
    assert TokenChunker(WhitespaceTokenizer(), 50, 5).chunk("   ") == []


def test_long_text_windows_overlap_and_cover_everything():
    ch = TokenChunker(WhitespaceTokenizer(), max_tokens=22, overlap=5)  # window 20, step 15
    chunks = ch.chunk(words(100))
    assert all(c.token_count <= 22 for c in chunks)
    seen = set()
    for c in chunks:
        seen.update(c.content.split())
    assert seen == set(words(100).split())
    a, b = chunks[0].content.split(), chunks[1].content.split()
    assert a[-5:] == b[:5]                               # 5-token overlap
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_last_chunk_adds_new_content():
    chunks = TokenChunker(WhitespaceTokenizer(), max_tokens=22, overlap=5).chunk(words(36))
    # windows start at 0, 15, 30 -> the last one holds w30..w35
    assert chunks[-1].content.split()[-1] == "w35"
    assert len(chunks) == 3


def test_title_reduces_budget():
    ch = TokenChunker(WhitespaceTokenizer(), max_tokens=22, overlap=5)
    with_title = ch.chunk(words(40), title="a three word")
    without = ch.chunk(words(40))
    assert len(with_title) >= len(without)
    assert all(c.token_count <= 22 for c in with_title)


def test_rejects_bad_overlap():
    with pytest.raises(ValueError):
        TokenChunker(WhitespaceTokenizer(), max_tokens=20, overlap=10)
