# Nano RAG

Fully on-device RAG (Retrieval-Augmented Generation) app for Android, built and tested on a
Pixel 10 Pro. No network calls at query time — embedding, retrieval, and generation all run
locally; the only thing that ever touches the network is a one-time embedder model download.

- **Embedding**: Google's [AI Edge RAG SDK](https://developers.google.com/edge/mediapipe/solutions/genai/rag/android) — the `Gecko-110m-en` embedder (Gemma-family, on-device).
- **Generation**: Gemini Nano via ML Kit's GenAI Prompt API (AICore) — same engine used in the `pixel-offline-assistant` comparison benchmark, not the RAG SDK's own MediaPipe/Gemma `.task` generation path.
- **Vector store**: `SqliteVectorStore` (AI Edge RAG SDK), persisted to internal storage — survives app restarts.
- **Keyword index**: an in-house BM25 index, persisted via a JSONL manifest and rebuilt in memory on launch.
- **Corpus**: bundled as APK assets (`app/src/main/assets/rag_docs/`, plaintext and PDF) — zero external-storage dependency (see gotcha below).

## Setup

Standard Gradle/Android build, no NDK:

```bash
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

On first launch the app downloads the Gecko embedder model (~443MB) directly from Hugging Face
into its own internal storage — keep the app in the foreground/screen on until that finishes. On
every launch after that, `RagPipelineFactory` detects the persisted vector store + chunk manifest
and skips extraction/chunking/embedding entirely.

## Features

- **Hybrid retrieval (BM25 + embeddings via RRF)** — `CompositeRetriever` fuses a keyword
  ranked list and a vector-similarity ranked list with Reciprocal Rank Fusion, so exact-term
  matches (model names, numbers) survive even when embedding similarity alone would rank them low.
- **HyDE (Hypothetical Document Embeddings)** — instead of embedding the raw question, Gemini
  Nano first drafts a plausible *answer* passage, and that passage gets embedded and searched
  instead. A short question and a real answer passage are phrased very differently, so a
  question's embedding often sits further from the relevant chunks than a fabricated (even if
  factually wrong) answer's embedding does. One extra Nano call per query, with a same-query
  fallback on any failure.
- **Query rewriting** — Nano turns a terse user query into a clearer standalone search query
  before retrieval, with the same fail-open fallback pattern used throughout the pipeline.
- **Lexical-overlap reranking** — blends vector/RRF score with plain term overlap (30% weight)
  as a cheap correction for what pure semantic similarity misses.
- **System 2 Attention (S2A) context filtering** — a second, single-purpose Nano call that
  extracts only the sentences relevant to the question from the retrieved chunks, before the
  chunks ever reach the answering call. Keeps irrelevant retrieved text from biasing the answer.
- **Confidence-gated answering** — if the best-ranked chunk's fused score falls below a
  threshold, or the S2A-refined context ends up empty, the pipeline skips the refine + generate
  calls entirely and returns a fixed "nothing relevant found" response, instead of letting Nano
  generate off of noise (and instead of spending two more on-device inference calls on a query
  that was never going anywhere).
- **Citations with tap-to-expand** — each answer is shown with its source chunks
  (`[filename #chunkIndex] (score)`); tapping a source expands the full chunk text inline.
- **Query embedding cache** — an LRU cache (64 entries) in front of `embedQuery`, so repeated
  or retried questions skip a full on-device embedder pass.
- **PDF ingestion** — PDFBox-Android extracts text from real-world, messy PDFs (broken
  ligatures, interleaved headers/footers) rather than requiring curated plaintext.
- **Persistent index across launches** — both the vector store and the BM25 manifest survive
  app restarts, so the bundled corpus is only ever indexed once.

## High-level design

Six-stage pipeline, each stage an injected interface (Strategy pattern), orchestrated by
`RagPipeline`:

```
                     ┌─────────────────────────── indexDocument() ───────────────────────────┐
                     │                                                                        │
   documents ──▶ PdfTextExtractor ──▶ TextChunker ──▶ EmbeddingService.embedDocument ──▶ VectorRepository
                                           │                                          └──▶ KeywordIndex
                                           └── ChunkMetadata (sourceId, chunkIndex)

                     ┌────────────────────────────── ask(query) ────────────────────────────────┐
                     ▼                                                                            
   query ──▶ QueryRewriter ──▶ Retriever (Composite: HyDE-embedding + BM25, fused via RRF) ──▶
              (Nano)                                                                              
                     ──▶ Reranker (lexical overlap) ──▶ [confidence gate] ──▶ ContextRefiner (S2A, Nano)
                                                                                      │
                                                                                      ▼
                                                                          AnswerGenerator (Nano)
                                                                                      │
                                                                                      ▼
                                                                                  AskResult
```

- **Stage 1 — QueryRewriter**: `NanoQueryRewriter` (Nano) / `PassthroughQueryRewriter` (no-op).
- **Stage 2 — Retriever**: `CompositeRetriever` fusing `HydeRetriever` (Nano-generated
  hypothetical answer → embedding → nearest-neighbor) and `KeywordRetriever` (BM25) via RRF.
  `EmbeddingRetriever` (plain query embedding, no HyDE) exists as a drop-in alternative.
- **Stage 3 — Reranker**: `LexicalOverlapReranker`.
- **Confidence gate** (in `RagPipeline` itself, not a separate stage): compares the
  top reranked score against a threshold before spending any more Nano calls.
- **Stage 4 — ContextRefiner**: `S2AContextRefiner` (Nano context-hygiene pass, falls back to
  `ConcatenatingContextRefiner`'s dedup+char-budget concatenation on failure).
- **Stage 5 — DocumentIndexer**: `ChunkingDocumentIndexer` — dual-writes each chunk into the
  vector store and the keyword index.
- **Stage 6 — AnswerGenerator**: `NanoAnswerGenerator`.

`RagPipelineFactory` wires the app's default stage mix; anything wanting a different combination
(no-op rewriter for speed, embedding-only retrieval, no HyDE) constructs a `RagPipeline` directly
instead of going through the factory.

## Low-level design notes

**Fail-open by default.** Every Nano-backed stage (`NanoQueryRewriter`, `HydeRetriever`,
`S2AContextRefiner`) wraps its model call in a try/catch and falls back to the pre-stage input
(raw query, raw query embedding, unfiltered context) on any failure or blank response. None of
these enrichment stages should ever be the reason a question goes unanswered.

**Confidence gate math.** `LexicalOverlapReranker` blends RRF-fused score (tiny — at most
`~1/(rrfK+1)` per contributing list, `rrfK=60`) with term-overlap fraction (`0..1`) at a 30%
lexical weight. In practice the blended score is dominated by lexical overlap; `RagPipeline`'s
default `minConfidenceScore = 0.05f` is calibrated to catch cases with near-zero term overlap
*and* weak RRF signal — i.e., queries where nothing in the corpus plausibly matches.

**Metadata survives fusion.** `CompositeRetriever` originally fused ranked lists keyed only by
chunk text, discarding `ChunkMetadata` in the process — silently breaking any UI (like source
citations) that needs to know which document a chunk came from. It now tracks the first
`RetrievedChunk` seen per text alongside the fused score and reattaches its metadata to the
result.

**Two persistence paths, one manifest.** The vector store (`SqliteVectorStore`) persists itself
natively. The BM25 index (`Bm25KeywordIndex`) is a rescore-the-whole-corpus-at-query-time
in-memory structure with nothing to persist directly — so `PersistedChunkStore` writes a
JSONL manifest of every `(text, metadata)` pair, and `PersistingKeywordIndex` decorates the base
index to replay writes into it. On a warm launch, the manifest is read back and replayed into a
fresh in-memory `Bm25KeywordIndex` — no re-embedding needed, since only the keyword index (not
the vector store) needs rebuilding from scratch each process.

**Query-embedding cache.** `CachingEmbeddingService` decorates any `EmbeddingService`,
LRU-caching (`LinkedHashMap` in access-order mode, capacity 64, `Mutex`-guarded for coroutine
safety) `embedQuery` results keyed by trimmed/lowercased text. `embedDocument` is never cached —
chunk text is effectively unique per corpus, so caching it would only cost memory for no hit rate.

**Chunking.** `SlidingWindowTextChunker` (default, 500 chars / 100 char overlap) avoids the hard
paragraph-boundary cuts of `ParagraphTextChunker`, so a sentence sitting on a chunk boundary still
appears whole in at least one chunk.

**Storage gotcha (why the model downloads at runtime instead of being adb-pushed).** On this
device/OS build, files written into the app's **external** files dir via `adb push` or
`adb shell mkdir` were invisible to the app's own `java.io.File` API at runtime —
`exists()`/`listFiles()` silently failed even though the files were genuinely there and
POSIX-readable via plain `adb shell`. `run-as <pkg>` confirmed it wasn't a race: the app's own UID
got `Permission denied` reading and writing there, on what turned out to be a production **"user"
build** (not userdebug/eng), which enforces scoped-storage isolation more strictly than adb
tooling assumes. Fix: skip external storage entirely — the embedder model downloads via plain
`HttpURLConnection` straight into internal storage (`filesDir`), and the document corpus ships as
bundled APK assets instead of being pushed separately.

## SDK API notes (undocumented at time of writing)

Google's official RAG guide doesn't show the exact `insert`/retrieval method signatures. We
extracted them directly from the real `localagents-rag-0.1.0.aar` via `javap -p` on its
`classes.jar` (see the parent conversation for the full technique) since the linked GitHub sample
repo (`google-ai-edge/ai-edge-apis`) currently 404s:

```java
// com.google.ai.edge.localagents.rag.models.Embedder<T>
ListenableFuture<ImmutableList<Float>> getEmbeddings(EmbeddingRequest<T>)
ListenableFuture<ImmutableList<ImmutableList<Float>>> getBatchEmbeddings(EmbeddingRequest<T>)

// com.google.ai.edge.localagents.rag.memory.VectorStore<T>
void insert(VectorStoreRecord<T>)
List<VectorStoreRecord<T>> getNearestRecords(List<Float> queryEmbedding, int topK, float minScore)

// GeckoEmbeddingModel constructor
GeckoEmbeddingModel(String modelPath, Optional<String> tokenizerPath, boolean useGpu)

// EmbedData.TaskType — asymmetric retrieval embeddings (index docs vs. embed the query differently)
RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, SEMANTIC_SIMILARITY, QUESTION_ANSWERING, ...
```

`DefaultVectorStore<T>` (in-memory, no persistence) and `SqliteVectorStore` (persistent) both
implement `VectorStore<T>` with the same two methods above.

## Package layout

```
rag/
├── RagPipeline.kt              orchestrator: rewrite → retrieve → rerank → [gate] → refine → generate
├── RagPipelineFactory.kt       wires the default stage implementations
├── chunking/                   TextChunker, SlidingWindowTextChunker, ParagraphTextChunker
├── embedding/                  EmbeddingService, GeckoEmbeddingService, CachingEmbeddingService
├── ingest/                     PdfTextExtractor
├── model/                      AskResult, RetrievedChunk, ChunkMetadata
├── pipeline/                   QueryRewriter / Retriever / Reranker / ContextRefiner /
│                               DocumentIndexer / AnswerGenerator interfaces + implementations
│                               (NanoQueryRewriter, HydeRetriever, EmbeddingRetriever,
│                               KeywordRetriever, CompositeRetriever, LexicalOverlapReranker,
│                               S2AContextRefiner, ConcatenatingContextRefiner,
│                               ChunkingDocumentIndexer, NanoAnswerGenerator)
└── store/                      VectorRepository, SqliteVectorRepository, InMemoryVectorRepository,
                                 KeywordIndex, Bm25KeywordIndex, PersistingKeywordIndex,
                                 PersistedChunkStore
ui/
└── RagActivity.kt               single-activity demo UI: model download, indexing progress,
                                  and a query box that shows every pipeline stage's output
                                  (rewritten query, retrieved/reranked chunks, refined context,
                                  answer, tap-to-expand source citations)
```
