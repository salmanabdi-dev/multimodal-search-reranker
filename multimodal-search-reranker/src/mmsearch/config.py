"""All tunable settings in one place.

Every value can be overridden with an environment variable prefixed ``MMS_``
(e.g. ``MMS_RERANK_CANDIDATES=20``). Evaluation and benchmark reports dump the
full settings object so every number is traceable to its configuration.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MMS_", env_file=".env", extra="ignore")

    # --- infrastructure ---
    database_url: str = "postgresql://mms:mms@localhost:5432/mms"
    db_pool_min: int = 2
    db_pool_max: int = 8
    log_level: str = "INFO"

    # --- compute ---
    device: str = "cpu"                 # "cpu" or "cuda"
    torch_threads: int = 4              # intra-op threads per forward pass
    inference_workers: int = 1          # model calls are serialised on this many threads

    # --- bi-encoder (stage 1) ---
    bi_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    bi_encoder_revision: str = "main"
    embedding_dim: int = 384            # must match vector(N) in sql/001_init.sql
    query_max_length: int = 64          # queries are short; smaller pad = faster
    doc_max_length: int = 256           # all-MiniLM-L6-v2 positional limit we respect
    query_prefix: str = ""              # e.g. "Represent this sentence for searching..." for BGE
    embed_batch_size: int = 64

    # --- chunking ---
    chunk_tokens: int = 256             # includes special tokens and the prepended title
    chunk_overlap: int = 32
    title_max_tokens: int = 32

    # --- vector retrieval ---
    candidate_chunks: int = 50          # K: chunks fetched from HNSW
    ef_search: int = 100                # HNSW query-time beam width (forced >= K)
    hnsw_m: int = 16
    hnsw_ef_construction: int = 128
    index_maintenance_work_mem: str = "1GB"

    # --- cross-encoder (stage 2) ---
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_revision: str = "main"
    rerank_candidates: int = 25         # N: documents scored by the cross-encoder
    rerank_max_length: int = 256
    rerank_quantize: bool = False       # int8 dynamic quantisation of Linear layers (CPU)

    # --- images (CLIP) ---
    enable_images: bool = True
    clip_model: str = "openai/clip-vit-base-patch32"
    clip_revision: str = "main"
    clip_dim: int = 512                 # must match vector(N) in sql/002_images.sql
    clip_text_max_length: int = 77      # CLIP's positional limit
    image_embed_batch_size: int = 16
    image_candidates: int = 20          # M: images fetched from HNSW and fused/reranked
    media_dir: str = "data/media"       # uploaded/ingested image files
    max_image_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 40_000_000  # decompression-bomb guard (~6300 x 6300)

    # --- per-modality calibration (score -> relevance probability) ---
    # p = sigmoid(a * score + b). Cross-encoder logits are already log-odds (a=1, b=0).
    # CLIP and bi-encoder cosines are not; defaults are rough and are meant to be fit
    # with `mmsearch calibrate-images` (CLIP) on the dev split.
    clip_calib_a: float = 40.0          # midpoint at cosine 0.25
    clip_calib_b: float = -10.0
    text_vec_calib_a: float = 12.0      # used only when reranking is off
    text_vec_calib_b: float = -6.0
    caption_weight: float = 0.5         # image relevance = w*caption_ce + (1-w)*clip

    # --- API ---
    default_top_k: int = 5
    max_top_k: int = 20
    max_query_chars: int = 1000

    # --- evaluation ---
    eval_seed: int = 13
    eval_dev_fraction: float = 0.4

    latency_target_ms: float = Field(65.0, description="p95 target used by the benchmark")

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.chunk_overlap >= self.chunk_tokens // 2:
            raise ValueError("chunk_overlap must be < chunk_tokens / 2")
        if not 0.0 <= self.caption_weight <= 1.0:
            raise ValueError("caption_weight must be in [0, 1]")
        if self.rerank_candidates < self.default_top_k:
            raise ValueError("rerank_candidates must be >= default_top_k")
        return self

    def k_chunks_for(self, n_docs: int) -> int:
        """Chunks to fetch so that, after collapsing to documents, ~n_docs remain."""
        return max(self.candidate_chunks, 2 * n_docs)

    def ef_for(self, k_chunks: int) -> int:
        """pgvector returns at most ef_search rows from an HNSW scan, so ef must be >= K."""
        return max(self.ef_search, k_chunks)


@lru_cache
def get_settings() -> Settings:
    return Settings()
