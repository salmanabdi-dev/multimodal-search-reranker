"""Evaluation data in BEIR layout:

    <dir>/corpus.jsonl      {"_id", "title", "text"}
    <dir>/queries.jsonl     {"_id", "text"}
    <dir>/qrels/<split>.tsv query-id <TAB> corpus-id <TAB> score   (with a header row)

The bundled sample set (data/sample) uses the same layout, so one loader serves both.
"""

from __future__ import annotations

import csv
import json
import logging
import random
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
BEIR_ROOT = PROJECT_ROOT / "data" / "beir"
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


@dataclass
class EvalDataset:
    name: str
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]      # query id -> {doc id: relevance grade}

    def relevant(self, qid: str, min_level: int = 1) -> set[str]:
        return {d for d, g in self.qrels.get(qid, {}).items() if g >= min_level}

    def judged_query_ids(self, min_level: int = 1) -> list[str]:
        """Queries that have text and at least one relevant document."""
        return sorted(q for q in self.qrels if q in self.queries and self.relevant(q, min_level))


def load_dataset(dataset_dir: str | Path, split: str = "test") -> EvalDataset:
    d = Path(dataset_dir)
    queries: dict[str, str] = {}
    with (d / "queries.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                obj = json.loads(line)
                queries[str(obj["_id"])] = obj["text"]

    qrels: dict[str, dict[str, int]] = {}
    with (d / "qrels" / f"{split}.tsv").open(encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)  # header
        for row in reader:
            if len(row) < 3:
                continue
            qid, did, score = row[0], row[1], int(float(row[2]))
            qrels.setdefault(qid, {})[did] = score
    return EvalDataset(name=d.name, queries=queries, qrels=qrels)


def split_query_ids(qids: list[str], dev_fraction: float, seed: int
                    ) -> tuple[list[str], list[str]]:
    """Deterministic dev/test split. Tune on dev, report on test, never the other way."""
    ids = sorted(qids)
    random.Random(seed).shuffle(ids)
    cut = int(round(len(ids) * dev_fraction))
    return sorted(ids[:cut]), sorted(ids[cut:])


def download_beir(name: str, dest_root: Path = BEIR_ROOT) -> Path:
    target = dest_root / name
    if (target / "corpus.jsonl").exists():
        log.info("%s already present at %s", name, target)
        return target
    dest_root.mkdir(parents=True, exist_ok=True)
    url = BEIR_URL.format(name=name)
    log.info("downloading %s", url)
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / f"{name}.zip"
        with urllib.request.urlopen(url, timeout=120) as resp, zpath.open("wb") as out:
            shutil.copyfileobj(resp, out)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(dest_root)
    return target
