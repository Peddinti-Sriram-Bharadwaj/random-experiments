package com.example.nanoassistant.rag.pipeline

import com.example.nanoassistant.rag.embedding.EmbeddingService
import com.example.nanoassistant.rag.model.RetrievedChunk
import com.example.nanoassistant.rag.store.VectorRepository
import com.google.mlkit.genai.prompt.GenerativeModel

/**
 * Hypothetical Document Embeddings (HyDE): instead of embedding the raw query, ask Gemini Nano
 * to write a plausible *answer* to it first, then embed that generated passage as a document
 * and search with that. A short question and a real answer passage phrase things very
 * differently, so a question's embedding often sits further from relevant documents than a
 * fabricated (even if factually wrong) answer's embedding does — HyDE trades one extra Nano
 * generation call per query for closing that gap.
 *
 * Falls back to embedding the raw query directly on any failure — same philosophy as
 * [NanoQueryRewriter]: this stage should never be the reason a question can't be answered.
 */
class HydeRetriever(
    private val model: GenerativeModel,
    private val embeddingService: EmbeddingService,
    private val vectorRepository: VectorRepository
) : Retriever {

    override suspend fun retrieve(query: String, topK: Int): List<RetrievedChunk> {
        val embedding = try {
            val prompt = "Write a short, plausible passage that would answer this question, " +
                "as if it came from a reference document. Reply with only the passage, no " +
                "explanation.\n\nQuestion: $query"
            val response = model.generateContent(prompt)
            val hypotheticalDocument = response.candidates.firstOrNull()?.text?.trim()
            if (hypotheticalDocument.isNullOrBlank()) {
                embeddingService.embedQuery(query)
            } else {
                embeddingService.embedDocument(hypotheticalDocument)
            }
        } catch (e: Exception) {
            embeddingService.embedQuery(query)
        }
        return vectorRepository.findNearest(embedding, topK)
    }
}
