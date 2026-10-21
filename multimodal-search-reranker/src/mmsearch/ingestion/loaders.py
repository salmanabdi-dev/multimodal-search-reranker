"""Document sources. Each loader yields ``RawDocument``s; nothing downstream cares where
they came from."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RawDocument:
    external_id: str
    text: str
    title: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def load_jsonl(path: str | Path, source: str | None = None) -> Iterator[RawDocument]:
    """One JSON object per line with an id (``_id``/``id``), ``text``, optional ``title``.

    This is also the BEIR ``corpus.jsonl`` format.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            doc_id = obj.get("_id") or obj.get("id") or obj.get("external_id")
            text = obj.get("text") or obj.get("content") or ""
            if doc_id is None:
                raise ValueError(f"{path}:{lineno}: missing document id")
            yield RawDocument(
                external_id=str(doc_id),
                text=text,
                title=obj.get("title") or None,
                source=source or path.stem,
                metadata=obj.get("metadata") or {},
            )


def load_directory(path: str | Path, extensions: tuple[str, ...] = (".txt", ".md")
                   ) -> Iterator[RawDocument]:
    """Every text/markdown file under ``path``; id = relative path, title = first heading."""
    root = Path(path)
    for file in sorted(p for p in root.rglob("*") if p.suffix.lower() in extensions):
        text = file.read_text(encoding="utf-8", errors="replace")
        title = file.stem.replace("_", " ").replace("-", " ")
        for line in text.splitlines():
            if line.startswith("#"):
                title = line.lstrip("#").strip()
                break
        yield RawDocument(external_id=str(file.relative_to(root)), text=text, title=title,
                          source=root.name)


def load_beir_corpus(dataset_dir: str | Path) -> Iterator[RawDocument]:
    dataset_dir = Path(dataset_dir)
    return load_jsonl(dataset_dir / "corpus.jsonl", source=dataset_dir.name)
