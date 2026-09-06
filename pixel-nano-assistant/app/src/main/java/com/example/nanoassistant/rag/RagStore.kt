package com.example.nanoassistant.rag

import com.google.ai.edge.localagents.rag.memory.DefaultVectorStore
import com.google.ai.edge.localagents.rag.memory.VectorStoreRecord
import com.google.ai.edge.localagents.rag.models.EmbedData
import com.google.ai.edge.localagents.rag.models.EmbeddingRequest
import com.google.ai.edge.localagents.rag.models.GeckoEmbeddingModel
import com.google.common.collect.ImmutableList
import kotlinx.coroutines.guava.await
import java.util.Optional

/**
 * Retrieval half of the RAG pipeline: Gecko (Gemma-family) embeddings + an in-memory
 * vector store. Generation is handled separately by NanoLlmEngine — this class only
 * turns a query into the most relevant chunks of previously-indexed text.
 */
class RagStore(geckoModelPath: String, tokenizerPath: String) {

    private val embedder = GeckoEmbeddingModel(geckoModelPath, Optional.of(tokenizerPath), false)
    private val store = DefaultVectorStore<String>()

    /** Splits on blank lines, then hard-wraps anything still too long for a single chunk. */
    fun chunk(text: String, maxChars: Int = 500): List<String> {
        val paragraphs = text.split(Regex("\n\\s*\n")).map { it.trim() }.filter { it.isNotEmpty() }
        val chunks = mutableListOf<String>()
        for (p in paragraphs) {
            if (p.length <= maxChars) {
                chunks += p
            } else {
                var start = 0
                while (start < p.length) {
                    val end = minOf(start + maxChars, p.length)
                    chunks += p.substring(start, end)
                    start = end
                }
            }
        }
        return chunks
    }

    suspend fun indexChunk(text: String) {
        val request = EmbeddingRequest.create(
            listOf(EmbedData.create(text, EmbedData.TaskType.RETRIEVAL_DOCUMENT))
        )
        val embedding: ImmutableList<Float> = embedder.getEmbeddings(request).await()
        store.insert(VectorStoreRecord.create(text, embedding))
    }

    suspend fun retrieve(query: String, topK: Int = 3): List<String> {
        val request = EmbeddingRequest.create(
            listOf(EmbedData.create(query, EmbedData.TaskType.RETRIEVAL_QUERY))
        )
        val queryEmbedding = embedder.getEmbeddings(request).await()
        return store.getNearestRecords(queryEmbedding, topK, 0.0f).map { it.data }
    }
}
