package com.example.nanoassistant.rag.store

import com.example.nanoassistant.rag.model.RetrievedChunk
import com.google.ai.edge.localagents.rag.memory.DefaultVectorStore
import com.google.ai.edge.localagents.rag.memory.VectorStoreRecord
import kotlin.math.sqrt

/** Adapts Google's in-memory [DefaultVectorStore] to our [VectorRepository] port, and computes
 *  cosine similarity ourselves since the SDK's nearest-neighbor call returns records, not scores. */
class InMemoryVectorRepository : VectorRepository {

    private val store = DefaultVectorStore<String>()

    override fun insert(text: String, embedding: List<Float>) {
        store.insert(VectorStoreRecord.create(text, com.google.common.collect.ImmutableList.copyOf(embedding)))
    }

    override fun findNearest(queryEmbedding: List<Float>, topK: Int): List<RetrievedChunk> =
        store.getNearestRecords(queryEmbedding, topK, 0.0f).map { record ->
            RetrievedChunk(record.data, cosineSimilarity(queryEmbedding, record.embeddings))
        }

    private fun cosineSimilarity(a: List<Float>, b: List<Float>): Float {
        var dot = 0f; var normA = 0f; var normB = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        val denom = sqrt(normA) * sqrt(normB)
        return if (denom == 0f) 0f else dot / denom
    }
}
