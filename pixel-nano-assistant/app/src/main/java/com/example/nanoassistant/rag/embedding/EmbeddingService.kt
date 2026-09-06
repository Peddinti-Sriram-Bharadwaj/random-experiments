package com.example.nanoassistant.rag.embedding

/** Turns text into a vector. Two methods because query/document embeddings are asymmetric
 *  for retrieval-tuned models (a query is embedded differently than the passage it should match). */
interface EmbeddingService {
    suspend fun embedDocument(text: String): List<Float>
    suspend fun embedQuery(text: String): List<Float>
}
