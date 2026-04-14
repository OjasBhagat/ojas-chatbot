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

---

## Problem Statement

Traditional RAG systems fail in two compounding ways:

1. **Chunking destroys context** — splitting a document into fixed-size pieces strips the surrounding meaning from each chunk. A chunk that says "the method returned 42" is useless without knowing what method or why.
2. **Semantic search alone misses exact terms** — dense vector embeddings find conceptually similar content, but fail on proper nouns, product codes, acronyms, and precise keyword matches.

**Goal:** Build a teaching-assistant chatbot that accepts PDFs or blog URLs, ingests them with context-preserving chunking, retrieves relevant content using both semantic and keyword signals, reranks for precision, and generates grounded answers through a clean Streamlit UI.

---

## Tech Stack

| Layer | Technology | Version | Why This Choice |
|---|---|---|---|
| **Document Loading** | LangChain `PyPDFLoader` + `WebBaseLoader` | `langchain>=0.2` | Unified interface for both PDF and URL sources; handles encoding, pagination, and metadata automatically |
| **Contextual Chunking** | LangChain `RecursiveCharacterTextSplitter` + Anthropic Claude | `langchain-text-splitters` | Recursive splitter respects sentence/paragraph boundaries; Claude enriches each chunk with a surrounding-context prefix |
| **Embeddings** | `sentence-transformers` — `all-MiniLM-L6-v2` | `sentence-transformers>=2.7` | Free, local inference; 384-dim vectors; excellent speed/quality trade-off for English Q&A tasks |
| **Vector Store** | ChromaDB (local persistent) | `chromadb>=0.5` | No Docker or cloud account required; persists to disk; supports metadata filtering and similarity search out of the box |
| **Keyword Search** | BM25 via `rank_bm25` | `rank-bm25>=0.2` | Industry-standard TF-IDF variant; handles stopwords, term frequency, and document length normalization |
| **Hybrid Fusion** | Reciprocal Rank Fusion (RRF) — custom implementation | — | Parameter-free; no training data needed; outperforms weighted score averaging on standard benchmarks |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `sentence-transformers>=2.7` | Cross-encoders attend to both query and passage together, giving far better relevance scores than bi-encoders |
| **LLM** | Anthropic Claude `claude-sonnet-4-6` | `anthropic>=0.30` | High instruction-following quality; prompt caching reduces cost on repeated context blocks |
| **UI** | Streamlit | `streamlit>=1.35` | Built-in `st.chat_message` / `st.chat_input` components; Python-native; minimal boilerplate |
| **Config / Secrets** | `python-dotenv` | `python-dotenv>=1.0` | Keeps `ANTHROPIC_API_KEY` out of source code |

---

## Folder Structure

```
d:\KICKDRUM\chatbot\
│
├── docs/
│   └── IMPLEMENTATION_PLAN.md          ← this document
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
│       └── app.py                      ← Streamlit entry point (run with: streamlit run src/ui/app.py)
│
├── data/
│   ├── raw/                            ← Uploaded PDFs stored here
│   └── chroma_db/                      ← ChromaDB persisted vector store
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_generation.py
│
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
║                                               │  (Claude API)                │   ║
║                                               │  Each chunk → Claude adds    │   ║
║                                               │  1–2 sentence context prefix │   ║
║                                               └────────────┬────────────────┘   ║
║                                                            │                    ║
║                                                            ▼                    ║
║                          ┌─────────────────────────────────────────────────┐   ║
║                          │            Embedding Generation                   │   ║
║                          │        (sentence-transformers MiniLM)             │   ║
║                          │          384-dim dense vectors                    │   ║
║                          └──────────────┬──────────────────────────────────┘   ║
║                                         │                                       ║
║                          ┌──────────────▼──────────────────────────────────┐   ║
║                          │              ChromaDB (local disk)               │   ║
║                          │    Stores: vector + raw text + metadata          │   ║
║                          │    Path: data/chroma_db/                         │   ║
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
This is a hands-on exercise with no production SLA. ChromaDB persists to a local folder, requires zero infrastructure, and provides the same query API as cloud alternatives. Migrating to Pinecone later is a one-file change.

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
- Contextual enrichment prompt should be: *"Here is the document section: {surrounding_text}. Here is the chunk to enrich: {chunk}. In 1–2 sentences, describe what this chunk is about in the context of the full document."*
- Batch Claude enrichment calls — use `asyncio.gather` to call multiple chunks concurrently rather than sequentially
- Validate that loaded text is not empty before chunking

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

### Phase Summary
The ingestion pipeline is a one-time operation per document. It transforms raw, unstructured source material into semantically self-sufficient text chunks. The key innovation is the contextual enrichment step — without it, retrieved chunks may be accurate matches but confusing in isolation. With it, every chunk can stand alone as a meaningful answer fragment. The output feeds directly into Phase 2.

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

### Definition of Done
- All enriched chunks from Phase 1 are embedded and stored in ChromaDB
- ChromaDB collection persists to disk and is queryable after app restart
- `query(text, top_k=20)` returns a list of `{text, metadata, score}` dicts
- Unit test: embed 5 test chunks, query with a related sentence, top result is the most semantically similar chunk

### Edge Cases
| Scenario | Handling |
|---|---|
| Chunk text exceeds model max token length (256 tokens for MiniLM) | Truncate `combined_text` to 256 tokens before embedding; log truncation |
| Same document ingested twice | Check for existing collection by source hash; skip re-embedding with a "already indexed" message |
| ChromaDB collection corrupted (e.g. incomplete write) | Catch `chromadb.errors.InvalidCollectionException`; delete and re-create collection |
| Zero chunks to embed (empty document) | Raise `EmbeddingError` before calling ChromaDB |
| Disk full during ChromaDB write | Surface `OSError` to UI with "Disk space insufficient" message |

### Phase Summary
Phase 2 converts the enriched text chunks into a mathematical representation (dense vectors) that enables similarity-based retrieval. ChromaDB acts as both the vector index and the document store — given a query vector, it returns the most geometrically similar chunk vectors along with their original text. This is the semantic half of the hybrid search system. The keyword half is built in Phase 3.

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
Phase 3 is the retrieval engine — the core of what makes this system more powerful than a standard RAG pipeline. Running semantic and keyword search in parallel captures both conceptually-related content and exact-term matches. The RRF fusion layer combines their ranked lists using a robust mathematical formula that requires no tuning. The output is a diverse, deduplicated set of 20 candidates that feed into the precision-focused reranking step.

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
- The cross-encoder model downloads automatically on first use to `~/.cache/huggingface/` — no manual setup needed

### Definition of Done
- `rerank(query, candidates, top_k=5)` returns an ordered list of 5 most relevant chunks
- Unit test: given 10 candidates with 2 clearly on-topic and 8 off-topic, top-2 results are the on-topic ones
- Cross-encoder scores are logged at DEBUG level

### Edge Cases
| Scenario | Handling |
|---|---|
| Fewer than 5 candidates (sparse corpus or short document) | Return all available candidates without padding |
| Cross-encoder model not downloaded | `sentence-transformers` auto-downloads; first call may take 30–60 seconds — show spinner in UI |
| Chunk text is very long (>512 tokens) | Cross-encoder tokenizer auto-truncates to 512 — acceptable; log a warning |
| All candidates have very low scores (query is off-topic) | Return top-5 regardless of score; the LLM system prompt instructs it to say "I don't know" |
| Duplicate chunk texts (different metadata, same content) | Both pass through reranking; LLM deduplicates semantically in generation |

### Phase Summary
Reranking is a precision filter. The retrieval stage (Phase 3) is optimized for recall — get everything plausibly relevant into the candidate set. Reranking throws away 15 of those 20 candidates, keeping only the 5 that most directly address the question. The cross-encoder's ability to read both the query and the passage simultaneously (rather than independently encoding them) gives it a significant accuracy advantage. After this phase, we have a small, high-quality context window ready for the LLM.

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
- Sidebar contains: file uploader (`st.file_uploader`), URL text input, "Index Document" button, index status indicator, "Clear Conversation" button
- Show a `st.spinner` during document ingestion and during LLM generation
- Display source citations as a collapsed `st.expander` below each assistant message
- On app start, check if ChromaDB already has indexed documents and show status

### Definition of Done
- User can upload a PDF or enter a URL and click "Index Document" — pipeline runs end to end without errors
- User can type a question and receive a streamed response with source citations
- Multi-turn conversation works (follow-up questions reference prior context via `st.session_state`)
- Prompt cache hits are visible in the Anthropic usage dashboard (input_tokens_cache_read > 0 on 2nd+ queries)
- App runs with: `streamlit run src/ui/app.py`

### Edge Cases
| Scenario | Handling |
|---|---|
| No documents indexed when user submits a query | Show `st.warning("Please index a document first")` and block submission |
| `ANTHROPIC_API_KEY` missing from `.env` | Catch `anthropic.AuthenticationError`; show `st.error` with setup instructions |
| Claude returns "I don't know" | Display as normal assistant message — not an error state |
| LLM API timeout (>30s) | Catch `anthropic.APITimeoutError`; show `st.error("Response timed out. Please try again.")` |
| User submits empty query | Streamlit `st.chat_input` ignores empty submissions natively |
| Browser refresh clears chat history | Expected behavior — `st.session_state` is per-session; ChromaDB persists, so re-querying works |
| Very long conversation (>50 turns) | Trim `st.session_state.messages` to last 20 turns before sending to Claude to avoid context overflow |
| PDF upload exceeds Streamlit default 200MB limit | Set `server.maxUploadSize=500` in `.streamlit/config.toml` or add file size validation |

### Phase Summary
Phase 5 is the user-facing layer that ties every upstream component together. The Streamlit UI provides the ingestion entry point (sidebar), manages session state across a conversation, and renders streamed responses in real time. The LLM layer uses prompt caching to make repeated queries cheap — after the first question, the system prompt and context block are served from Anthropic's cache, making follow-up answers both faster and more cost-efficient. The source citation block gives users verifiable grounding for every answer.

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
