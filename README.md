Multimodal Search & Reranker

A multimodal semantic search engine combining natural-language document retrieval, image search, and neural reranking to improve search relevance and retrieval efficiency.

Overview

The system uses a two-stage retrieval architecture to identify and rank relevant content from natural-language queries. It integrates transformer-based embeddings, vector similarity search, and cross-encoder reranking to support efficient retrieval across text and image content.

Tech Stack

Languages: Python, SQL

AI/ML: PyTorch, Hugging Face Transformers, CLIP, Bi-Encoders, Cross-Encoders

Backend: FastAPI

Database: PostgreSQL, pgvector

Infrastructure: Docker

Indexing: HNSW (Hierarchical Navigable Small World)

Key Features

Semantic Document Search: Retrieves relevant documents using transformer-based vector embeddings.

Multimodal Retrieval: Supports text and image search through CLIP embeddings.

Two-Stage Retrieval: Combines approximate nearest-neighbor vector search with cross-encoder reranking.

Vector Indexing: Uses pgvector HNSW indexes for efficient similarity search.

REST API: Provides endpoints for document ingestion, image ingestion, and search.

Performance Evaluation: Includes Precision@5 evaluation and latency benchmarking.

Automated Optimization: Evaluates retrieval configurations to balance search quality and response time.

Containerization: Uses Docker for reproducible application and database setup.

Architecture

Document Retrieval

Natural Language Query → Bi-Encoder → Query Embedding → pgvector HNSW Search → Cross-Encoder Reranking → Top-5 Results

Image Retrieval

Natural Language Query → CLIP Text Encoder → Vector Similarity Search → Relevant Images

Performance

The project includes benchmarking and evaluation tools for measuring:

Search latency (mean, p50, p95, p99)

Precision@5 retrieval quality

Improvements from cross-encoder reranking

Text, image, and combined search performance

Performance targets include sub-65ms search latency and a 28% improvement in Top-5 retrieval precision. Final results are subject to benchmark verification.

Project Status

Core implementation completed. End-to-end Docker verification, integration testing, and performance benchmarking are pending.

Technologies

Python PyTorch FastAPI PostgreSQL pgvector Docker Hugging Face CLIP HNSW
