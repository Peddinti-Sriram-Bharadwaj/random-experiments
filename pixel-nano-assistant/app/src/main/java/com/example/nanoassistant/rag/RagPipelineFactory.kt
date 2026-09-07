package com.example.nanoassistant.rag

import android.content.Context
import com.example.nanoassistant.rag.chunking.SlidingWindowTextChunker
import com.example.nanoassistant.rag.embedding.GeckoEmbeddingService
import com.example.nanoassistant.rag.pipeline.ChunkingDocumentIndexer
import com.example.nanoassistant.rag.pipeline.CompositeRetriever
import com.example.nanoassistant.rag.pipeline.ConcatenatingContextRefiner
import com.example.nanoassistant.rag.pipeline.EmbeddingRetriever
import com.example.nanoassistant.rag.pipeline.KeywordRetriever
import com.example.nanoassistant.rag.pipeline.LexicalOverlapReranker
import com.example.nanoassistant.rag.pipeline.NanoAnswerGenerator
import com.example.nanoassistant.rag.pipeline.NanoQueryRewriter
import com.example.nanoassistant.rag.pipeline.S2AContextRefiner
import com.example.nanoassistant.rag.store.Bm25KeywordIndex
import com.example.nanoassistant.rag.store.PersistedChunkStore
import com.example.nanoassistant.rag.store.PersistingKeywordIndex
import com.example.nanoassistant.rag.store.SqliteVectorRepository
import com.google.mlkit.genai.prompt.Generation
import java.io.File

/**
 * Builds a [RagPipeline] wired with this app's default stage implementations: Gecko embeddings +
 * BM25 keyword search fused via Reciprocal Rank Fusion for retrieval, Gemini Nano for
 * rewrite/generate. Callers who want a different mix — a no-op rewriter for speed, embedding-only
 * retrieval — construct a [RagPipeline] directly instead; this factory only covers the common
 * case.
 *
 * Both the vector store and the keyword index are persistent across launches (see
 * [SqliteVectorRepository] and [PersistedChunkStore]/[PersistingKeywordIndex]) — [alreadyIndexed]
 * tells the caller whether a previous launch already populated them, so it can skip re-running
 * extraction/chunking/embedding over the bundled docs entirely.
 */
object RagPipelineFactory {

    data class Result(val pipeline: RagPipeline, val alreadyIndexed: Boolean)

    suspend fun create(
        context: Context,
        geckoModelPath: String,
        geckoTokenizerPath: String
    ): Result {
        val embeddingService = GeckoEmbeddingService(geckoModelPath, geckoTokenizerPath)
        val chunker = SlidingWindowTextChunker()
        val chunkStore = PersistedChunkStore(context)
        val alreadyIndexed = chunkStore.exists()

        if (!alreadyIndexed) {
            // No manifest means either a first run or a previous run that never finished —
            // either way, don't reopen a stale/partial vectors.db alongside a fresh BM25 index.
            File(context.filesDir, "vectors.db").delete()
        }

        // The vector store needs its embedding dimension up front (not just on first insert) so
        // that reopening an existing on-disk database — the whole point of skipping re-indexing
        // — actually works.
        val embeddingDimension = embeddingService.embedQuery("dimension probe").size
        val vectorRepository = SqliteVectorRepository(context, embeddingDimension)

        val baseKeywordIndex = Bm25KeywordIndex()
        if (alreadyIndexed) {
            chunkStore.readAll().forEach { (text, metadata) -> baseKeywordIndex.insert(text, metadata) }
        }
        val keywordIndex = PersistingKeywordIndex(baseKeywordIndex, chunkStore)
        val nanoModel = Generation.getClient()

        val retriever = CompositeRetriever(
            listOf(
                EmbeddingRetriever(embeddingService, vectorRepository),
                KeywordRetriever(baseKeywordIndex)
            )
        )

        val pipeline = RagPipeline(
            queryRewriter = NanoQueryRewriter(nanoModel),
            retriever = retriever,
            reranker = LexicalOverlapReranker(),
            contextRefiner = S2AContextRefiner(nanoModel, fallback = ConcatenatingContextRefiner()),
            documentIndexer = ChunkingDocumentIndexer(chunker, embeddingService, vectorRepository, keywordIndex),
            answerGenerator = NanoAnswerGenerator(nanoModel)
        )
        return Result(pipeline, alreadyIndexed)
    }
}
