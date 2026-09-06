package com.example.nanoassistant.rag

import com.example.nanoassistant.rag.model.AskResult
import com.example.nanoassistant.rag.pipeline.AnswerGenerator
import com.example.nanoassistant.rag.pipeline.ContextRefiner
import com.example.nanoassistant.rag.pipeline.DocumentIndexer
import com.example.nanoassistant.rag.pipeline.QueryRewriter
import com.example.nanoassistant.rag.pipeline.Reranker
import com.example.nanoassistant.rag.pipeline.Retriever

/**
 * Facade over the six-stage RAG flow: rewrite -> retrieve -> rerank -> refine -> generate,
 * plus the separate insert (indexing) path chunk/retrieve/rerank/refine/generate don't touch.
 *
 * Every stage is injected as an interface (Strategy pattern) — swapping, say, [Reranker] for a
 * cross-encoder-based one later means writing one new class, not touching this orchestrator
 * (Open/Closed). See [com.example.nanoassistant.rag.RagPipelineFactory] for how a default
 * pipeline is wired.
 */
class RagPipeline(
    private val queryRewriter: QueryRewriter,
    private val retriever: Retriever,
    private val reranker: Reranker,
    private val contextRefiner: ContextRefiner,
    private val documentIndexer: DocumentIndexer,
    private val answerGenerator: AnswerGenerator
) {
    suspend fun indexDocument(text: String, sourceId: String, onChunkIndexed: (Int) -> Unit = {}) {
        documentIndexer.index(text, sourceId, onChunkIndexed)
    }

    suspend fun ask(query: String, topK: Int = 3): AskResult {
        val rewrittenQuery = queryRewriter.rewrite(query)
        val retrieved = retriever.retrieve(rewrittenQuery, topK)
        val reranked = reranker.rerank(rewrittenQuery, retrieved)
        val context = contextRefiner.refine(rewrittenQuery, reranked)
        val answer = answerGenerator.generate(rewrittenQuery, context)

        return AskResult(
            originalQuery = query,
            rewrittenQuery = rewrittenQuery,
            retrievedChunks = retrieved,
            rerankedChunks = reranked,
            refinedContext = context,
            answer = answer
        )
    }
}
