package com.example.nanoassistant.rag

import com.example.nanoassistant.rag.model.AskResult
import com.example.nanoassistant.rag.pipeline.AnswerGenerator
import com.example.nanoassistant.rag.pipeline.ContextRefiner
import com.example.nanoassistant.rag.pipeline.DocumentIndexer
import com.example.nanoassistant.rag.pipeline.QueryRewriter
import com.example.nanoassistant.rag.pipeline.Reranker
import com.example.nanoassistant.rag.pipeline.Retriever
import com.example.nanoassistant.rag.security.PromptInjectionGuard

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
    private val answerGenerator: AnswerGenerator,
    private val minConfidenceScore: Float = 0.05f
) {
    suspend fun indexDocument(text: String, sourceId: String, onChunkIndexed: (Int) -> Unit = {}) {
        documentIndexer.index(text, sourceId, onChunkIndexed)
    }

    suspend fun ask(query: String, topK: Int = 3): AskResult {
        val rewrittenQuery = queryRewriter.rewrite(query)
        val retrieved = retriever.retrieve(rewrittenQuery, topK)
        val reranked = reranker.rerank(rewrittenQuery, retrieved)

        // Prompt-injection defense: a chunk can come from *any* indexed document, including ones
        // an attacker planted, so it's untrusted input the moment it's about to flow into a
        // generation prompt — drop chunks matching known instruction-hijacking idioms before they
        // ever reach the context refiner or answer generator. Deliberately runs before the
        // confidence gate too, so a malicious chunk can't win the gate on a high lexical-overlap
        // score just because it echoes the query terms back convincingly.
        val (guarded, flaggedCount) = PromptInjectionGuard.scan(reranked)

        // Confidence gate: if even the best-ranked (post-guard) chunk barely matches the query,
        // skip the context-refine + generate calls entirely (saves two Nano round trips on a
        // device where that matters) rather than risk Nano confidently answering off of noise.
        val topScore = guarded.firstOrNull()?.score ?: 0f
        val lowConfidence = topScore < minConfidenceScore

        val context = if (lowConfidence) "" else contextRefiner.refine(rewrittenQuery, guarded)
        val answer = if (lowConfidence || context.isBlank()) {
            LOW_CONFIDENCE_ANSWER
        } else {
            answerGenerator.generate(rewrittenQuery, context)
        }

        return AskResult(
            originalQuery = query,
            rewrittenQuery = rewrittenQuery,
            retrievedChunks = retrieved,
            rerankedChunks = guarded,
            refinedContext = context,
            answer = answer,
            isLowConfidence = lowConfidence || context.isBlank(),
            blockedByInjectionGuard = flaggedCount
        )
    }

    private companion object {
        const val LOW_CONFIDENCE_ANSWER =
            "I couldn't find anything in your documents that looks relevant to that question."
    }
}
