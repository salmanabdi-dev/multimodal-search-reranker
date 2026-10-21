"""Token-window chunking using the bi-encoder's own tokenizer.

Why tokens rather than characters: the model truncates at a fixed number of *tokens*.
Counting with the same tokenizer guarantees no part of a chunk is silently cut off
and therefore invisible to search.

Windows overlap so a sentence straddling a boundary appears whole in at least one
chunk. The token budget per window reserves room for the special tokens
([CLS]/[SEP]) and for the title, which is prepended to every chunk (see
``passage_text``). The budget is approximate at the title/body seam; the model's
own truncation is the safety net.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class TokenizerLike(Protocol):
    special_tokens: int

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Character span (start, end) of every token, without special tokens."""
        ...


@dataclass(frozen=True)
class Chunk:
    index: int
    content: str
    token_count: int       # tokens the model will see, including title and specials


class TokenChunker:
    def __init__(self, tokenizer: TokenizerLike, max_tokens: int = 256, overlap: int = 32,
                 title_max_tokens: int = 32, min_window: int = 32) -> None:
        if overlap < 0 or overlap >= max_tokens // 2:
            raise ValueError("overlap must be in [0, max_tokens/2)")
        self.tok = tokenizer
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.title_max_tokens = title_max_tokens
        self.min_window = min_window

    def chunk(self, text: str, title: str | None = None) -> list[Chunk]:
        text = text.strip()
        if not text:
            return []

        body_budget = self.max_tokens - self.tok.special_tokens
        title_cost = 0
        if title:
            title_cost = min(len(self.tok.offsets(title)), self.title_max_tokens) + 1  # + "."
            # A long title may not squeeze the body below min_window; the title gives way
            # (the model truncates it) rather than the chunk exceeding max_tokens.
            title_cost = max(0, min(title_cost, body_budget - min(self.min_window, body_budget)))
        window = body_budget - title_cost
        overlap = min(self.overlap, window // 2)
        step = window - overlap

        offs = self.tok.offsets(text)
        if not offs:
            return []

        chunks: list[Chunk] = []
        start = 0
        while True:
            end = min(start + window, len(offs))
            content = text[offs[start][0]:offs[end - 1][1]].strip()
            chunks.append(Chunk(index=len(chunks), content=content,
                                token_count=(end - start) + self.tok.special_tokens + title_cost))
            if end == len(offs):
                return chunks
            start += step
