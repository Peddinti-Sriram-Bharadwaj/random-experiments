package com.example.nanoassistant.rag

import com.example.nanoassistant.rag.chunking.SemanticTextChunker
import com.example.nanoassistant.rag.chunking.SlidingWindowTextChunker
import com.example.nanoassistant.rag.chunking.TextChunker
import com.example.nanoassistant.rag.embedding.EmbeddingService
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
import com.example.nanoassistant.rag.store.InMemoryVectorRepository
import com.google.mlkit.genai.prompt.Generation

/**
 * Builds a [RagPipeline] wired with this app's default stage implementations: Gecko embeddings +
 * BM25 keyword search fused via Reciprocal Rank Fusion for retrieval, Gemini Nano for
 * rewrite/generate. Callers who want a different mix — a persistent vector store, a no-op
 * rewriter for speed, embedding-only retrieval — construct a [RagPipeline] directly instead;
 * this factory only covers the common case.
 */
object RagPipelineFactory {

    enum class ChunkingStrategy { SLIDING_WINDOW, SEMANTIC }

    fun create(
        geckoModelPath: String,
        geckoTokenizerPath: String,
        chunkingStrategy: ChunkingStrategy = ChunkingStrategy.SLIDING_WINDOW
    ): RagPipeline {
        val embeddingService = GeckoEmbeddingService(geckoModelPath, geckoTokenizerPath)
        val chunker = chunkerFor(chunkingStrategy, embeddingService)
        val vectorRepository = InMemoryVectorRepository()
        val keywordIndex = Bm25KeywordIndex()
        val nanoModel = Generation.getClient()

        val retriever = CompositeRetriever(
            listOf(
                EmbeddingRetriever(embeddingService, vectorRepository),
                KeywordRetriever(keywordIndex)
            )
        )

        return RagPipeline(
            queryRewriter = NanoQueryRewriter(nanoModel),
            retriever = retriever,
            reranker = LexicalOverlapReranker(),
            contextRefiner = S2AContextRefiner(nanoModel, fallback = ConcatenatingContextRefiner()),
            documentIndexer = ChunkingDocumentIndexer(chunker, embeddingService, vectorRepository, keywordIndex),
            answerGenerator = NanoAnswerGenerator(nanoModel)
        )
    }

    private fun chunkerFor(strategy: ChunkingStrategy, embeddingService: EmbeddingService): TextChunker =
        when (strategy) {
            ChunkingStrategy.SLIDING_WINDOW -> SlidingWindowTextChunker()
            ChunkingStrategy.SEMANTIC -> SemanticTextChunker(embeddingService)
        }
}
