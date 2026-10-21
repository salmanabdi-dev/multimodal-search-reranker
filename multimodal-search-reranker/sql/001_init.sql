-- Schema for the main database. Applied automatically by the postgres container on
-- first start, and by `mmsearch init-db` otherwise. Idempotent.
--
-- The HNSW index is NOT created here: the ingestion pipeline builds it after a bulk
-- load (much faster than inserting row by row into an existing graph).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id            BIGSERIAL PRIMARY KEY,
    external_id   TEXT NOT NULL UNIQUE,          -- stable ID from the source; qrels refer to it
    title         TEXT,
    source        TEXT,
    modality      TEXT NOT NULL DEFAULT 'text',  -- reserved for a future image encoder
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id               BIGSERIAL PRIMARY KEY,
    document_id      BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index      INT NOT NULL,
    content          TEXT NOT NULL,
    token_count      INT NOT NULL,
    embedding        vector(384) NOT NULL,
    embedding_model  TEXT NOT NULL,               -- guards against mixing vector spaces
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
