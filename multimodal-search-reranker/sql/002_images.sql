-- Images (migration 002). Idempotent; applied on container first start and by init_db,
-- which the API also runs at start-up, so existing volumes get upgraded in place.
--
-- An image is a row in `documents` (modality = 'image') plus one row here. CLIP vectors
-- are 512-d, MiniLM's are 384-d, so they cannot share the chunks.embedding column.
CREATE TABLE IF NOT EXISTS image_embeddings (
    document_id      BIGINT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    sha256           TEXT NOT NULL UNIQUE,        -- content address: identical bytes = one image
    file_path        TEXT NOT NULL,               -- relative to MMS_MEDIA_DIR
    mime_type        TEXT NOT NULL,
    width            INT NOT NULL,
    height           INT NOT NULL,
    bytes            INT NOT NULL,
    caption          TEXT,                        -- optional; enables caption reranking
    embedding        vector(512) NOT NULL,
    embedding_model  TEXT NOT NULL
);

-- Unlike the text index (rebuilt after bulk loads), images arrive one upload at a time,
-- so this index is created up front and maintained incrementally on insert.
CREATE INDEX IF NOT EXISTS image_embeddings_hnsw
    ON image_embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

CREATE INDEX IF NOT EXISTS documents_modality_idx ON documents (modality);
