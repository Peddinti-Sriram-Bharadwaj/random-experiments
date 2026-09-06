package com.example.nanoassistant.rag.pipeline

import com.example.nanoassistant.rag.embedding.EmbeddingService
import com.example.nanoassistant.rag.model.RetrievedChunk
import com.example.nanoassistant.rag.store.VectorRepository

/** Embeds the query and does a nearest-neighbor lookup against the vector store. */
class EmbeddingRetriever(
    private val embeddingService: EmbeddingService,
    private val vectorRepository: VectorRepository
) : Retriever {

    override suspend fun retrieve(query: String, topK: Int): List<RetrievedChunk> {
        val queryEmbedding = embeddingService.embedQuery(query)
        return vectorRepository.findNearest(queryEmbedding, topK)
    }
}
