package com.example.nanoassistant.rag

import com.example.nanoassistant.rag.chunking.ParagraphTextChunker
import com.example.nanoassistant.rag.embedding.GeckoEmbeddingService
import com.example.nanoassistant.rag.pipeline.ChunkingDocumentIndexer
import com.example.nanoassistant.rag.pipeline.ConcatenatingContextRefiner
import com.example.nanoassistant.rag.pipeline.EmbeddingRetriever
import com.example.nanoassistant.rag.pipeline.LexicalOverlapReranker
import com.example.nanoassistant.rag.pipeline.NanoAnswerGenerator
import com.example.nanoassistant.rag.pipeline.NanoQueryRewriter
import com.example.nanoassistant.rag.store.InMemoryVectorRepository
import com.google.mlkit.genai.prompt.Generation

/**
 * Builds a [RagPipeline] wired with this app's default stage implementations (Gecko embeddings,
 * in-memory vector store, Gemini Nano for rewrite/generate). Callers who want a different mix —
 * a persistent vector store, a no-op rewriter for speed, a different reranker — construct a
 * [RagPipeline] directly instead; this factory only covers the common case.
 */
object RagPipelineFactory {

    fun create(geckoModelPath: String, geckoTokenizerPath: String): RagPipeline {
        val embeddingService = GeckoEmbeddingService(geckoModelPath, geckoTokenizerPath)
        val vectorRepository = InMemoryVectorRepository()
        val nanoModel = Generation.getClient()

        return RagPipeline(
            queryRewriter = NanoQueryRewriter(nanoModel),
            retriever = EmbeddingRetriever(embeddingService, vectorRepository),
            reranker = LexicalOverlapReranker(),
            contextRefiner = ConcatenatingContextRefiner(),
            documentIndexer = ChunkingDocumentIndexer(ParagraphTextChunker(), embeddingService, vectorRepository),
            answerGenerator = NanoAnswerGenerator(nanoModel)
        )
    }
}
