# Implementation Plan: Context-Aware Hybrid-Search RAG Chatbot

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Tech Stack](#tech-stack)
3. [Folder Structure](#folder-structure)
4. [Architecture Diagram](#architecture-diagram)
5. [Architecture Explanation](#architecture-explanation)
6. [Phase 1 — Document Ingestion & Contextual Chunking](#phase-1--document-ingestion--contextual-chunking)
7. [Phase 2 — Vector Embeddings & Storage](#phase-2--vector-embeddings--storage)
8. [Phase 3 — Hybrid Search (Semantic + Keyword)](#phase-3--hybrid-search-semantic--keyword)
9. [Phase 4 — Reranking](#phase-4--reranking)
10. [Phase 5 — LLM Integration & Streamlit UI](#phase-5--llm-integration--streamlit-ui)
11. [Containerization](#containerization)

---

## Problem Statement

Traditional RAG systems fail in two compounding ways:

1. **Chunking destroys context** — splitting a document into fixed-size pieces strips the surrounding meaning from each chunk. A chunk that says "the method returned 42" is useless without knowing what method or why.
2. **Semantic search alone misses exact terms** — dense vector embeddings find conceptually similar content, but fail on proper nouns, product codes, acronyms, and precise keyword matches.

**Goal:** Build a teaching-assistant chatbot that accepts PDFs or blog URLs, ingests them with context-preserving chunking, retrieves relevant content using both semantic and keyword signals, reranks for precision, and generates grounded answers through a clean Streamlit UI — packaged as a single Docker container so it runs identically on any machine.

---

## Tech Stack

| Layer | Technology | Version | Why This Choice |
|---|---|---|---|
| **Document Loading** | LangChain `PyPDFLoader` + `WebBaseLoader` | `langchain>=0.2` | Unified interface for both PDF and URL sources; handles encoding, pagination, and metadata automatically |
| **Contextual Chunking** | LangChain `RecursiveCharacterTextSplitter` + Anthropic Claude | `langchain-text-splitters` | Recursive splitter respects sentence/paragraph boundaries; Claude enriches each chunk with a surrounding-context prefix |
| **Embeddings** | `sentence-transformers` — `all-MiniLM-L6-v2` | `sentence-transformers>=2.7` | Free, local inference; 384-dim vectors; excellent speed/quality trade-off for English Q&A tasks |
| **Vector Store** | ChromaDB (local persistent) | `chromadb>=0.5` | No external service required; persists to disk via volume mount; same query API as cloud alternatives |
| **Keyword Search** | BM25 via `rank_bm25` | `rank-bm25>=0.2` | Industry-standard TF-IDF variant; handles stopwords, term frequency, and document length normalization |
| **Hybrid Fusion** | Reciprocal Rank Fusion (RRF) — custom implementation | — | Parameter-free; no training data needed; outperforms weighted score averaging on standard benchmarks |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `sentence-transformers>=2.7` | Cross-encoders attend to both query and passage together, giving far better relevance scores than bi-encoders |
| **LLM** | Anthropic Claude `claude-sonnet-4-6` | `anthropic>=0.30` | High instruction-following quality; prompt caching reduces cost on repeated context blocks |
| **UI** | Streamlit | `streamlit>=1.35` | Built-in `st.chat_message` / `st.chat_input` components; Python-native; minimal boilerplate |
| **Config / Secrets** | `python-dotenv` | `python-dotenv>=1.0` | Keeps `ANTHROPIC_API_KEY` out of source code; works with Docker `env_file` directive |
| **Containerization** | Docker + Docker Compose | `docker>=25`, `compose>=2.24` | Reproducible single-command deployment; volume mounts persist data across rebuilds |

---

## Folder Structure

```
d:\KICKDRUM\chatbot\
│
├── docs/
│   └── implementation_plan_claude.md   ← this document
│
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py                   ← PDF + URL document loaders
│   │   └── chunker.py                  ← Recursive splitting + Claude contextual enrichment
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedder.py                 ← Embedding generation + ChromaDB read/write
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── semantic_search.py          ← ChromaDB vector similarity query
│   │   ├── keyword_search.py           ← BM25 index build + query
│   │   └── hybrid_search.py            ← RRF fusion of semantic + keyword results
│   │
│   ├── reranking/
│   │   ├── __init__.py
│   │   └── reranker.py                 ← Cross-encoder scoring + top-K selection
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   └── llm.py                      ← Claude API call with prompt caching + streaming
│   │
│   └── ui/
│       └── app.py                      ← Streamlit entry point
│
├── data/
│   ├── raw/                            ← Uploaded PDFs (mounted as Docker volume)
│   └── chroma_db/                      ← ChromaDB persisted vector store (mounted as Docker volume)
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_generation.py
│
├── .streamlit/
│   └── config.toml                     ← Streamlit server config (port, upload size limits)
│
├── Dockerfile                          ← Single-stage container build
├── docker-compose.yml                  ← Orchestration: service, volumes, env, ports
├── .dockerignore                       ← Excludes data/, .env, __pycache__, .git
├── .env                                ← ANTHROPIC_API_KEY=sk-ant-...  (never commit)
├── .env.example                        ← ANTHROPIC_API_KEY=your_key_here
├── requirements.txt
└── README.md
```

---

## Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                         INGESTION PIPELINE (run once per document)              ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║   ┌──────────────┐    ┌──────────────────┐    ┌────────────────────────────┐    ║
║   │  PDF / URL   │───▶│  Document Loader  │───▶│  Recursive Text Splitter   │    ║
║   │  (user input)│    │  (LangChain)      │    │  chunk_size=512            │    ║
║   └──────────────┘    │  PyPDFLoader /    │    │  chunk_overlap=128         │    ║
║                        │  WebBaseLoader   │    └────────────┬───────────────┘    ║
║                        └──────────────────┘                 │                    ║
║                                                             ▼                    ║
║                                               ┌─────────────────────────────┐   ║
║                                               │  Contextual Enrichment       │   ║
║                                               │  (Claude API — external)     │   ║
║                                               │  Each chunk → Claude adds    │   ║
║                                               │  1–2 sentence context prefix │   ║
║                                               └────────────┬────────────────┘   ║
║                                                            │                    ║
║                                                            ▼                    ║
║                          ┌─────────────────────────────────────────────────┐   ║
║                          │            Embedding Generation                   │   ║
║                          │        (sentence-transformers MiniLM)             │   ║
║                          │          384-dim dense vectors (local)            │   ║
║                          └──────────────┬──────────────────────────────────┘   ║
║                                         │                                       ║
║                          ┌──────────────▼──────────────────────────────────┐   ║
║                          │      ChromaDB (embedded — Docker volume)         │   ║
║                          │    Stores: vector + raw text + metadata          │   ║
║                          │    Mount: ./data/chroma_db:/app/data/chroma_db   │   ║
║                          └─────────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════════╗
║                         QUERY PIPELINE (runs on every user message)             ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║   ┌─────────────────┐                                                           ║
║   │   User Query     │                                                           ║
║   └────────┬─────────┘                                                          ║
║            │                                                                     ║
║      ┌─────┴──────┐                                                             ║
║      │            │                                                              ║
║      ▼            ▼                                                              ║
║  ┌─────────┐  ┌──────────────┐                                                  ║
║  │ Semantic │  │  BM25 Keyword│                                                  ║
║  │  Search  │  │   Search     │                                                  ║
║  │(ChromaDB)│  │ (rank_bm25)  │                                                  ║
║  │  top-20  │  │   top-20     │                                                  ║
║  └────┬─────┘  └──────┬───────┘                                                 ║
║       │               │                                                          ║
║       └───────┬───────┘                                                          ║
║               ▼                                                                  ║
║   ┌───────────────────────┐                                                      ║
║   │  Reciprocal Rank       │                                                     ║
║   │  Fusion (RRF)          │                                                     ║
║   │  score = Σ 1/(k+rank)  │                                                     ║
║   │  k=60, dedup by id     │                                                     ║
║   │  → top-20 fused        │                                                     ║
║   └────────────┬──────────┘                                                      ║
║                ▼                                                                  ║
║   ┌───────────────────────────┐                                                  ║
║   │  Cross-Encoder Reranker    │                                                  ║
║   │  ms-marco-MiniLM-L-6-v2   │                                                  ║
║   │  Scores (query, chunk)     │                                                  ║
║   │  pairs → selects top-5     │                                                  ║
║   └────────────┬──────────────┘                                                  ║
║                ▼                                                                  ║
║   ┌───────────────────────────────────────────────────┐                          ║
║   │              Claude claude-sonnet-4-6              │                          ║
║   │                                                   │                          ║
║   │  System prompt (cached):                          │                          ║
║   │  "Answer only from the provided context.          │                          ║
║   │   Cite sources. Say I don't know if not found."   │                          ║
║   │                                                   │                          ║
║   │  Context block (cached):                          │                          ║
║   │  [top-5 reranked chunks with source metadata]     │                          ║
║   │                                                   │                          ║
║   │  User message: {query}                            │                          ║
║   └────────────┬──────────────────────────────────────┘                          ║
║                ▼                                                                  ║
║   ┌───────────────────────────┐                                                  ║
║   │  Streamed Answer + Sources │                                                  ║
║   │  → Streamlit Chat UI       │                                                  ║
║   └───────────────────────────┘                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## Architecture Explanation

### Why Contextual Chunking?
Standard splitting creates orphaned chunks — fragments that are syntactically complete but semantically incomplete. Contextual chunking solves this by making a Claude API call for each chunk, asking the model to read the chunk plus its surrounding chunks and prepend a 1–2 sentence context summary. The result: every chunk is self-contained and retrievable even when the question uses different vocabulary than the chunk itself.

### Why Hybrid Search?
No single search strategy covers all query types:
- **Semantic search** (ChromaDB + embeddings) excels at paraphrase matching — "what is the capital of France?" finds "Paris is the seat of the French government" even with zero overlapping words.
- **BM25 keyword search** excels at exact-term matching — model names, error codes, version numbers, names of people. Embeddings tend to embed these as generic concepts, losing precision.
- **Together via RRF**, the system has both high recall and high precision across query types.

### Why Reciprocal Rank Fusion (RRF)?
The naive alternative — weighted score averaging — requires calibrating two different score distributions (cosine similarity vs. BM25 score). RRF sidesteps this entirely by working on ranks, not scores. The formula `1/(k + rank)` is robust, parameter-free, and has been shown to match or beat tuned alternatives on the BEIR benchmark.

### Why Cross-Encoder Reranking?
The retrieval stage is optimized for recall (get everything relevant into the candidate set). The reranking stage is optimized for precision (pick the 5 chunks that most directly answer the question). Cross-encoders are slower but far more accurate than bi-encoders because they process query and passage together, allowing full attention between both. Running them on only 20 candidates keeps latency acceptable.

### Why Prompt Caching?
When a user asks multiple questions about the same document, the system prompt and the retrieved context blocks don't change between turns. Anthropic's prompt caching lets us mark these blocks with `cache_control: ephemeral`, reducing token cost by up to 90% on cache hits and cutting latency significantly on follow-up questions.

### Why ChromaDB Local (Not Pinecone/Weaviate)?
ChromaDB persists to a local folder mounted as a Docker volume. It requires zero external services, survives container restarts, and provides the same query API as cloud alternatives. Migrating to Pinecone later is a one-file change.

### Why a Single Container (Not Microservices)?
This is a hands-on exercise. A single `python:3.11-slim` container houses the entire application — Streamlit UI, RAG pipeline, ChromaDB embedded store, and local ML models. The only external dependency is the Anthropic API. This keeps `docker-compose up` a one-command deployment with no service mesh, no inter-container networking, and no operational overhead.

---

## Phase 1 — Document Ingestion & Contextual Chunking

### What Needs to Be Done
1. Accept a file path (PDF) or a URL from the user
2. Load the raw document text using the appropriate LangChain loader
3. Split the text into overlapping chunks using `RecursiveCharacterTextSplitter`
4. For each chunk, call Claude with the chunk + its 2 neighbors to generate a 1–2 sentence context prefix
5. Return enriched chunk objects with metadata (source, page number, chunk index)

### Tech Stack
| Tool | Purpose |
|---|---|
| `langchain.document_loaders.PyPDFLoader` | Load PDF files page by page |
| `langchain.document_loaders.WebBaseLoader` | Scrape and clean blog/article URLs |
| `langchain.text_splitter.RecursiveCharacterTextSplitter` | Sentence-aware splitting with overlap |
| `anthropic` SDK | Claude API call for contextual enrichment |

### Best Practices
- **chunk_size = 512 tokens, chunk_overlap = 128 tokens** — 25% overlap is the standard that balances context preservation with storage efficiency
- Always store `source` (filename or URL), `page` (for PDFs), and `chunk_index` in chunk metadata
- Contextual enrichment prompt: *"Here is the document section: {surrounding_text}. Here is the chunk to enrich: {chunk}. In 1–2 sentences, describe what this chunk is about in the context of the full document."*
- Batch Claude enrichment calls — use `asyncio.gather` to call multiple chunks concurrently rather than sequentially
- Validate that loaded text is not empty before chunking

### Containerization in This Phase
- The `ANTHROPIC_API_KEY` used by the enrichment Claude call is injected at container runtime via the `env_file: .env` directive in `docker-compose.yml` — no hardcoded credentials in the image
- Uploaded PDFs are saved to `/app/data/raw/` inside the container, which is mapped to `./data/raw/` on the host via a bind mount — files persist even if the container is stopped or rebuilt
- No code changes are needed between local and container execution; `PyPDFLoader` and `WebBaseLoader` work the same inside the container

### Definition of Done
- Given any valid PDF path or URL, the pipeline returns a list of `EnrichedChunk` objects
- Each chunk has: `text` (original), `context_prefix` (Claude-generated), `combined_text` (prefix + text), `metadata` dict
- Unit test passes: loading a known 2-page PDF produces ≥ 3 chunks with non-empty context prefixes

### Edge Cases
| Scenario | Handling |
|---|---|
| PDF with empty/image-only pages | Skip pages where extracted text is < 50 characters; log a warning |
| URL returns 403 / paywall | Catch `requests.HTTPError`; raise `DocumentLoadError` with user-friendly message |
| URL returns non-article HTML (homepage, login page) | Validate that extracted text length > 500 chars before proceeding |
| Document with < 2 chunks after splitting | Skip contextual enrichment (not enough context to enrich); use raw chunk text |
| Claude API rate limit during enrichment | Implement exponential backoff (1s, 2s, 4s) with max 3 retries |
| Non-UTF-8 encoded PDF | Use `errors='replace'` in text decoding; log character replacement count |
| Container has no network access | Claude API call will fail; surface as `ConnectionError` in UI with message to check network/proxy settings |

### Phase Summary
The ingestion pipeline is a one-time operation per document. It transforms raw, unstructured source material into semantically self-sufficient text chunks. The key innovation is the contextual enrichment step — without it, retrieved chunks may be accurate matches but confusing in isolation. With it, every chunk can stand alone as a meaningful answer fragment. Inside the container, uploaded files land on a host-mounted volume so they survive restarts. The output feeds directly into Phase 2.

---

## Phase 2 — Vector Embeddings & Storage

### What Needs to Be Done
1. Load the `all-MiniLM-L6-v2` model from `sentence-transformers`
2. Generate 384-dimensional embeddings for the `combined_text` of each enriched chunk
3. Initialize a persistent ChromaDB collection named after the document source
4. Store embeddings, original texts, and metadata in ChromaDB
5. Expose a query interface that accepts a query string and returns top-N similar chunks

### Tech Stack
| Tool | Purpose |
|---|---|
| `sentence-transformers` | Local embedding model inference |
| `chromadb` | Local persistent vector store |

### Best Practices
- **Embed `combined_text`** (context prefix + chunk), not just the raw chunk — the context prefix improves retrieval
- **Batch embedding** with `batch_size=64` to keep memory usage flat regardless of document size
- **Persistent storage** at `data/chroma_db/` so documents don't need to be re-embedded on every app restart
- **Deduplication**: before embedding, compute an MD5 hash of the document source. If a ChromaDB collection with that hash already exists, skip re-embedding
- Use **cosine similarity** (ChromaDB default) rather than Euclidean distance for text embeddings
- Store the original `text` (without prefix) as the ChromaDB document, and `combined_text` as metadata — this ensures the LLM sees clean context, not the synthetic prefix

### Containerization in This Phase
- ChromaDB writes to `/app/data/chroma_db/` inside the container, which is mapped to `./data/chroma_db/` on the host — the vector store persists across `docker stop`, `docker restart`, and even `docker-compose down` (data is on the host, not in the container layer)
- The HuggingFace model (`all-MiniLM-L6-v2`) downloads to `/root/.cache/huggingface/` on first run. This path is mounted as a named Docker volume (`huggingface_cache`) so the model is not re-downloaded on every container start
- To pre-bake the model into the image (eliminating the first-run download delay), add `RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"` to the Dockerfile after `pip install`

### Definition of Done
- All enriched chunks from Phase 1 are embedded and stored in ChromaDB
- ChromaDB collection persists to disk and is queryable after app restart
- `query(text, top_k=20)` returns a list of `{text, metadata, score}` dicts
- Unit test: embed 5 test chunks, query with a related sentence, top result is the most semantically similar chunk
- After `docker-compose down && docker-compose up`, previously indexed documents are still queryable (volume persistence test)

### Edge Cases
| Scenario | Handling |
|---|---|
| Chunk text exceeds model max token length (256 tokens for MiniLM) | Truncate `combined_text` to 256 tokens before embedding; log truncation |
| Same document ingested twice | Check for existing collection by source hash; skip re-embedding with an "already indexed" message |
| ChromaDB collection corrupted (e.g. incomplete write) | Catch `chromadb.errors.InvalidCollectionException`; delete and re-create collection |
| Zero chunks to embed (empty document) | Raise `EmbeddingError` before calling ChromaDB |
| Disk full during ChromaDB write | Surface `OSError` to UI with "Disk space insufficient" message |
| Docker volume mount is read-only (misconfiguration) | ChromaDB will throw `PermissionError`; surface as "Storage is read-only — check Docker volume permissions" |

### Phase Summary
Phase 2 converts the enriched text chunks into a mathematical representation (dense vectors) that enables similarity-based retrieval. ChromaDB acts as both the vector index and the document store. The critical container detail here is the two-volume strategy: the `chroma_db` bind mount ensures the vector store outlives the container; the `huggingface_cache` named volume ensures model weights are downloaded once and reused. This is the semantic half of the hybrid search system. The keyword half is built in Phase 3.

---

## Phase 3 — Hybrid Search (Semantic + Keyword)

### What Needs to Be Done
1. **Semantic Search**: Embed the user's query with MiniLM → query ChromaDB for top-20 similar chunks
2. **Keyword Search**: Build a BM25 index from all chunk texts → query for top-20 BM25-scored chunks
3. **Fusion**: Apply Reciprocal Rank Fusion to merge and re-rank the two result lists
4. Return a deduplicated list of top-20 fused results with unified scores

### Tech Stack
| Tool | Purpose |
|---|---|
| `chromadb` query API | Semantic search via cosine similarity |
| `rank_bm25.BM25Okapi` | Keyword-based BM25 retrieval |
| Custom RRF function | Score fusion without hyperparameter tuning |

### Best Practices
- **Retrieve top-20 from each arm** before fusion — wider recall at this stage is better; reranking will filter to 5
- **RRF formula**: for each document d appearing in results from search arm s: `score(d) += 1 / (k + rank_s(d))`, where `k=60` (standard value from original RRF paper)
- **BM25 index rebuilds in memory** at query time from the full chunk corpus — acceptable for <50,000 chunks; no persistence needed
- **Tokenization for BM25**: lowercase + split on whitespace (no stemming needed for general English Q&A)
- **Deduplication**: use `chunk_id` as the dedup key after fusion; keep the higher combined score
- Preserve original `text` and `metadata` on fused results for downstream reranking

### Containerization in This Phase
- BM25 runs entirely in-memory — no files written, no volumes needed
- ChromaDB queries read from the already-mounted `/app/data/chroma_db/` volume — no additional container configuration required
- The entire retrieval module is stateless from a container perspective: it reads from the volume, computes in RAM, and passes results downstream

### Definition of Done
- `hybrid_search(query, top_k=20)` returns a list of chunks combining both search signals
- Semantic-only queries (paraphrase questions) return better results than BM25 alone
- Exact-keyword queries (model names, codes) return better results than semantic alone
- Integration test: a query with an exact product name appears in fused top-5 but not in semantic-only top-5

### Edge Cases
| Scenario | Handling |
|---|---|
| Query matches nothing in BM25 (e.g. all stopwords) | BM25 returns empty list; RRF uses only semantic results — graceful degradation |
| Query is a single character | BM25 returns low-confidence scores; semantic search dominates — acceptable behavior |
| Corpus is empty (no documents indexed) | Check ChromaDB collection size before querying; return empty list with warning |
| Both search arms return the same top result | Dedup preserves one entry with combined score; no issues |
| Very short chunks (< 10 words) | BM25 naturally scores these low; semantic search handles them normally |
| Query longer than 256 tokens | Truncate query embedding input; BM25 handles full query natively |

### Phase Summary
Phase 3 is the retrieval engine — the core of what makes this system more powerful than a standard RAG pipeline. Running semantic and keyword search in parallel captures both conceptually-related content and exact-term matches. From a container standpoint, this phase is the simplest: it is fully stateless, reads from an already-mounted volume, and has no infrastructure dependencies. The RRF fusion layer combines ranked lists using a robust mathematical formula that requires no tuning. The output is a diverse, deduplicated set of 20 candidates that feed into the precision-focused reranking step.

---

## Phase 4 — Reranking

### What Needs to Be Done
1. Accept the 20 fused candidates from Phase 3
2. Build (query, chunk_text) pairs for each candidate
3. Run all pairs through the cross-encoder in a single batch
4. Sort by cross-encoder score (descending)
5. Return the top-5 chunks with scores and metadata

### Tech Stack
| Tool | Purpose |
|---|---|
| `sentence-transformers` `CrossEncoder` | `ms-marco-MiniLM-L-6-v2` relevance scoring |

### Best Practices
- **Always batch** all pairs in a single `cross_encoder.predict(pairs)` call — avoid per-chunk calls which add significant overhead
- **Input format**: `[query, chunk_text]` pairs — do not include the context prefix in the chunk text passed to the cross-encoder (it may confuse scoring)
- **top_k = 5** — enough context for Claude, small enough to stay well within the 200K token window
- Log cross-encoder scores in debug mode for observability and pipeline tuning
- Pre-bake the cross-encoder model into the Docker image to eliminate first-run download delay

### Containerization in This Phase
- Like the embedding model, the cross-encoder (`ms-marco-MiniLM-L-6-v2`) downloads to `/root/.cache/huggingface/` — the same `huggingface_cache` named volume covers both models, so neither re-downloads on container restart
- To pre-bake both models into the image, add to the Dockerfile after `pip install`:
  ```dockerfile
  RUN python -c "\
    from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('all-MiniLM-L6-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
  ```
  This makes the image larger (~400MB extra) but eliminates all model download time at container startup
- Reranking is CPU-bound. If running on a machine with a GPU, set `TRANSFORMERS_DEVICE=cuda` as an environment variable in `docker-compose.yml` and add `--gpus all` to the service definition for significant speedup

### Definition of Done
- `rerank(query, candidates, top_k=5)` returns an ordered list of 5 most relevant chunks
- Unit test: given 10 candidates with 2 clearly on-topic and 8 off-topic, top-2 results are the on-topic ones
- Cross-encoder scores are logged at DEBUG level
- Container startup shows no model download (models served from `huggingface_cache` volume or baked into image)

### Edge Cases
| Scenario | Handling |
|---|---|
| Fewer than 5 candidates (sparse corpus or short document) | Return all available candidates without padding |
| Cross-encoder model not downloaded and no internet in container | Fail fast with `OSError: model not found` — pre-baking the model in the image is the mitigation |
| Chunk text is very long (>512 tokens) | Cross-encoder tokenizer auto-truncates to 512 — acceptable; log a warning |
| All candidates have very low scores (query is off-topic) | Return top-5 regardless of score; the LLM system prompt instructs it to say "I don't know" |
| Duplicate chunk texts (different metadata, same content) | Both pass through reranking; LLM deduplicates semantically in generation |

### Phase Summary
Reranking is a precision filter. The retrieval stage (Phase 3) is optimized for recall — get everything plausibly relevant into the candidate set. Reranking throws away 15 of those 20 candidates, keeping only the 5 that most directly address the question. In the container context, the key decision is whether to pre-bake model weights into the image (larger image, fast startup) or rely on the named volume cache (smaller image, slow first start). Pre-baking is recommended for any deployment where startup latency matters. After this phase, we have a small, high-quality context window ready for the LLM.

---

## Phase 5 — LLM Integration & Streamlit UI

### What Needs to Be Done
1. Format the top-5 reranked chunks into a structured context block with source citations
2. Call Claude `claude-sonnet-4-6` with prompt caching enabled on the system prompt + context block
3. Stream the response token-by-token into the Streamlit UI
4. Build the Streamlit app: sidebar for document ingestion, main panel for chat, session state for history
5. Display source citations below each answer

### Tech Stack
| Tool | Purpose |
|---|---|
| `anthropic` SDK | Claude API with streaming and prompt caching |
| `streamlit` | Chat UI with `st.chat_message`, `st.chat_input`, `st.session_state` |

### Best Practices

**LLM Layer**
- **System prompt** (mark as `cache_control: ephemeral`):
  ```
  You are a teaching assistant. Answer questions based ONLY on the provided context.
  Cite the source document and page number for every claim.
  If the answer is not in the context, say "I don't have enough information to answer that."
  ```
- **Context block** (mark as `cache_control: ephemeral`): Format as numbered chunks with source metadata
- **Prompt caching**: cache both the system prompt and the context block — on follow-up questions in the same session, these cache hits reduce cost by ~80% and latency by ~50%
- **Streaming**: use `client.messages.stream()` and yield tokens as they arrive — do not wait for full response

**UI Layer**
- `st.session_state.messages` stores the full chat history as `[{"role": "user"/"assistant", "content": "..."}]`
- Sidebar: file uploader (`st.file_uploader`), URL text input, "Index Document" button, index status indicator, "Clear Conversation" button
- Show `st.spinner` during document ingestion and LLM generation
- Display source citations as a collapsed `st.expander` below each assistant message
- On app start, check if ChromaDB already has indexed documents and show status

### Containerization in This Phase
- Streamlit listens on `0.0.0.0:8501` inside the container (configured in `.streamlit/config.toml` and the `CMD` in the Dockerfile) — Docker maps this to `localhost:8501` on the host via the `ports: - "8501:8501"` directive in `docker-compose.yml`
- The `ANTHROPIC_API_KEY` is read from the container environment (injected via `env_file: .env`) — `python-dotenv` loads it with `load_dotenv()` at startup
- File uploads via `st.file_uploader` are written to `/app/data/raw/` which is bind-mounted to `./data/raw/` on the host
- `.streamlit/config.toml` should set `server.maxUploadSize = 500` to allow PDFs up to 500MB; this file is copied into the image via the Dockerfile
- Streamlit session state (`st.session_state`) is in-memory per browser session — it does not persist across container restarts (expected and acceptable)

### Definition of Done
- User can upload a PDF or enter a URL and click "Index Document" — pipeline runs end to end without errors
- User can type a question and receive a streamed response with source citations
- Multi-turn conversation works (follow-up questions reference prior context via `st.session_state`)
- Prompt cache hits are visible in the Anthropic usage dashboard (`input_tokens_cache_read > 0` on 2nd+ queries)
- App is accessible at `http://localhost:8501` after `docker-compose up`

### Edge Cases
| Scenario | Handling |
|---|---|
| No documents indexed when user submits a query | Show `st.warning("Please index a document first")` and block submission |
| `ANTHROPIC_API_KEY` missing from `.env` | Catch `anthropic.AuthenticationError`; show `st.error` with setup instructions |
| Claude returns "I don't know" | Display as normal assistant message — not an error state |
| LLM API timeout (>30s) | Catch `anthropic.APITimeoutError`; show `st.error("Response timed out. Please try again.")` |
| User submits empty query | Streamlit `st.chat_input` ignores empty submissions natively |
| Browser refresh clears chat history | Expected — `st.session_state` is per-session; ChromaDB persists on volume, so re-querying works |
| Very long conversation (>50 turns) | Trim `st.session_state.messages` to last 20 turns before sending to Claude to avoid context overflow |
| PDF upload exceeds Streamlit limit | Configurable in `.streamlit/config.toml`; default set to 500MB in this project |
| Container port 8501 already in use on host | `docker-compose up` will fail with `bind: address already in use`; change host port to `"8502:8501"` in `docker-compose.yml` |

### Phase Summary
Phase 5 is the user-facing layer that ties every upstream component together. The Streamlit UI provides the ingestion entry point (sidebar), manages session state across a conversation, and renders streamed responses in real time. Inside the container, the app is a long-running process bound to `0.0.0.0:8501` — Docker's port mapping makes it reachable from the host browser. The LLM layer uses prompt caching to make repeated queries cheap. The source citation block gives users verifiable grounding for every answer.

---

## Containerization

### Overview
The entire application is packaged as a single Docker container. There are no microservices. The container runs the Streamlit application, executes the full RAG pipeline in-process, and persists all stateful data (vector store, uploaded files, ML model weights) to the host filesystem via volume mounts. The only network egress required is outbound HTTPS to the Anthropic API.

### Container Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            Docker Host (your machine)                    │
│                                                                          │
│   browser                                                                │
│   http://localhost:8501 ──────────────────────────────────────────────┐ │
│                                                                        │ │
│  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │          chatbot  (python:3.11-slim)     port 8501              │  │ │
│  │                                                                  │◀─┘ │
│  │  ┌──────────────┐  ┌────────────────┐  ┌─────────────────────┐ │    │
│  │  │  Streamlit    │  │  RAG Pipeline  │  │  ChromaDB embedded  │ │    │
│  │  │  src/ui/app.py│─▶│  src/          │─▶│  /app/data/chroma_db│ │    │
│  │  │              │  │  ingestion/    │  │                     │ │    │
│  │  │  :8501       │  │  embeddings/   │  └──────────┬──────────┘ │    │
│  │  └──────────────┘  │  retrieval/    │             │            │    │
│  │                     │  reranking/    │             │ bind mount │    │
│  │  ┌──────────────┐  │  generation/   │             ▼            │    │
│  │  │  .env        │  └────────────────┘  ./data/chroma_db/       │    │
│  │  │  (env_file)  │                                               │    │
│  │  │  API KEY ────┼──────────────────────────────▶ Anthropic API │    │
│  │  └──────────────┘                                (HTTPS egress) │    │
│  │                                                                  │    │
│  │  ┌──────────────────────────────────────────────────────────┐  │    │
│  │  │              /root/.cache/huggingface/                    │  │    │
│  │  │  all-MiniLM-L6-v2  +  ms-marco-MiniLM-L-6-v2            │  │    │
│  │  │  (named volume: huggingface_cache)                        │  │    │
│  │  └──────────────────────────────────────────────────────────┘  │    │
│  │                                                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│   Host Volumes (persist data across container lifecycle):               │
│   ┌─────────────────────┐  ┌─────────────────────┐                     │
│   │  ./data/chroma_db/  │  │    ./data/raw/       │                     │
│   │  (vector store)     │  │  (uploaded PDFs)     │                     │
│   └─────────────────────┘  └─────────────────────┘                     │
│                                                                          │
│   Named Volume (Docker managed):                                        │
│   ┌─────────────────────────────────────────────┐                      │
│   │  huggingface_cache  (ML model weights)       │                      │
│   └─────────────────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why This Container Architecture?

| Decision | Rationale |
|---|---|
| **Single container** | No inter-service networking complexity; ChromaDB is an embedded library, not a server |
| **`python:3.11-slim` base** | Minimal attack surface; ~150MB smaller than `python:3.11`; all required packages install cleanly |
| **Bind mounts for data** | `./data/chroma_db/` and `./data/raw/` on the host — user can inspect, backup, or delete data without entering the container |
| **Named volume for model cache** | HuggingFace models are large and version-stable; a named volume keeps them across rebuilds without cluttering the project directory |
| **`env_file` for secrets** | `.env` is never copied into the image; it is injected at runtime only — secrets stay out of image layers |
| **Pre-baking ML models** | Optional but recommended: baking models into the image trades image size for zero first-run download latency |

### Dockerfile

```dockerfile
FROM python:3.11-slim

# Install system build dependencies (needed for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake ML models into the image (eliminates first-run download; ~400MB added to image)
RUN python -c "\
    from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('all-MiniLM-L6-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy application source
COPY src/ ./src/
COPY .streamlit/ ./.streamlit/
COPY .env.example .

# Create data directories (will be overridden by volume mounts at runtime)
RUN mkdir -p data/raw data/chroma_db

# Expose Streamlit port
EXPOSE 8501

# Health check — Streamlit has a built-in health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

### docker-compose.yml

```yaml
version: "3.9"

services:
  chatbot:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: rag_chatbot
    ports:
      - "8501:8501"          # host:container — access at http://localhost:8501
    env_file:
      - .env                 # injects ANTHROPIC_API_KEY at runtime (never baked into image)
    volumes:
      # Bind mounts: data persists on host, survives container rebuild
      - ./data/chroma_db:/app/data/chroma_db
      - ./data/raw:/app/data/raw
      # Named volume: ML model weights cached across container restarts
      - huggingface_cache:/root/.cache/huggingface
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

volumes:
  huggingface_cache:
    driver: local
```

### .dockerignore

```
# Data (mounted as volumes at runtime — never bake into image)
data/

# Secrets
.env

# Python artifacts
__pycache__/
*.pyc
*.pyo
.pytest_cache/
*.egg-info/

# Dev tools
.git/
.gitignore
tests/
docs/

# OS artifacts
.DS_Store
Thumbd.db
```

### .streamlit/config.toml

```toml
[server]
port = 8501
address = "0.0.0.0"
headless = true
maxUploadSize = 500        # MB — allows large PDFs

[browser]
gatherUsageStats = false
```

### Docker Compose Commands

| Command | Purpose |
|---|---|
| `docker-compose up --build` | Build image and start container (first run) |
| `docker-compose up` | Start container using existing image |
| `docker-compose up -d` | Start in detached (background) mode |
| `docker-compose down` | Stop and remove container (data volumes preserved) |
| `docker-compose down -v` | Stop and remove container **and all volumes** (destructive — deletes ChromaDB data) |
| `docker-compose logs -f` | Stream container logs |
| `docker-compose exec chatbot bash` | Open a shell inside the running container |
| `docker-compose build --no-cache` | Force full image rebuild (e.g., after changing requirements.txt) |

### Best Practices
- Never copy `.env` into the image — always use `env_file` in `docker-compose.yml`
- Pin the base image to a specific patch version in production (`python:3.11.9-slim`) for reproducibility
- Use `.dockerignore` aggressively — `data/` and `.git/` alone can save hundreds of MB from the build context
- The `HEALTHCHECK` directive enables `docker-compose` to report container health; the `--start-period=60s` accounts for model loading time on first start when models are not pre-baked
- Run `docker-compose config` before deploying to validate the compose file syntax

### Definition of Done
- `docker-compose up --build` completes without errors
- `http://localhost:8501` loads the Streamlit UI in a browser
- Indexing a document inside the container writes files to `./data/chroma_db/` on the host
- `docker-compose down && docker-compose up` — UI loads, previously indexed documents are still queryable
- `docker-compose down -v && docker-compose up --build` — clean slate, UI loads fresh

### Edge Cases

| Scenario | Handling |
|---|---|
| Port 8501 already in use on host | Change `ports` in `docker-compose.yml` to `"8502:8501"` |
| `.env` file missing | `docker-compose up` starts but app immediately shows `AuthenticationError`; create `.env` from `.env.example` |
| `docker-compose down -v` accidentally run | All ChromaDB data lost; must re-index documents. Mitigate with a periodic `cp -r ./data/chroma_db ./data/chroma_db.backup` |
| Model pre-bake step fails during build (no internet) | Remove the `RUN python -c "..."` pre-bake lines from Dockerfile; models will download on first container start using the `huggingface_cache` volume |
| Container runs out of memory during embedding | Set `mem_limit: 4g` under the service in `docker-compose.yml`; reduce `batch_size` in `embedder.py` to 16 |
| Windows host with WSL2 Docker backend | Volume bind mounts work correctly; use `./data/chroma_db` (forward slashes) in `docker-compose.yml` — Docker Desktop translates paths automatically |
| Running on ARM (Apple Silicon M-series) | `python:3.11-slim` has native ARM builds; `sentence-transformers` and `chromadb` support ARM — no changes needed |

---

## requirements.txt

```
# Document loading
langchain>=0.2
langchain-community>=0.2
langchain-text-splitters>=0.2
pypdf>=4.0

# Embeddings & vector store
sentence-transformers>=2.7
chromadb>=0.5

# Keyword search
rank-bm25>=0.2

# LLM
anthropic>=0.30

# UI
streamlit>=1.35

# Utilities
python-dotenv>=1.0
beautifulsoup4>=4.12
requests>=2.31
```

---

## End-to-End Verification Checklist

**Local (without Docker)**
- [ ] `pip install -r requirements.txt` completes without errors
- [ ] `.env` file created with valid `ANTHROPIC_API_KEY`
- [ ] `streamlit run src/ui/app.py` launches without import errors
- [ ] Upload a PDF → "Index Document" → status shows "X chunks indexed"
- [ ] Ask a question from the document → streamed response appears with source citations
- [ ] Ask a follow-up question → response references prior context
- [ ] Ask an out-of-scope question → Claude responds "I don't have enough information"
- [ ] Enter a blog URL → index → query → receive accurate answer
- [ ] Restart the app → ChromaDB still has the indexed document (persistence test)
- [ ] Check Anthropic usage dashboard — cache reads appear on 2nd+ queries (prompt caching test)

**Docker**
- [ ] `docker-compose up --build` completes without errors
- [ ] `http://localhost:8501` loads Streamlit UI in browser
- [ ] Index a document inside the container → `./data/chroma_db/` folder appears on host
- [ ] `docker-compose down && docker-compose up` → previously indexed document still queryable
- [ ] `docker-compose logs -f` shows no error-level log lines during normal operation
- [ ] `docker stats rag_chatbot` — memory usage stays under 3GB during embedding
- [ ] `docker-compose down -v && docker-compose up --build` → clean start, UI loads fresh with no stale data
