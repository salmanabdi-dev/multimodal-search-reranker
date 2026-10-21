"""Text normalisation, and the single definition of what text a model sees for a chunk."""

from __future__ import annotations

import re
import unicodedata

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_INLINE_WS = re.compile(r"[ \t\u00a0\u2000-\u200b]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """NFC-normalise, drop control characters, collapse whitespace, keep paragraphs."""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    lines = [_INLINE_WS.sub(" ", line).strip() for line in text.split("\n")]
    return _MANY_NEWLINES.sub("\n\n", "\n".join(lines)).strip()


def passage_text(title: str | None, content: str) -> str:
    """Text embedded at ingestion AND scored by the reranker. Keep these identical.

    The title is prepended to every chunk: a chunk from the middle of a document
    often doesn't repeat what the document is about, and the title supplies it.
    """
    if not title:
        return content
    sep = " " if title[-1] in ".!?:" else ". "
    return f"{title}{sep}{content}"
