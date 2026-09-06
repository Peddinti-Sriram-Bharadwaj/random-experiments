package com.example.nanoassistant.rag.pipeline

import com.example.nanoassistant.rag.model.RetrievedChunk

/** Stage 2: fetch candidate chunks for a (rewritten) query. */
interface Retriever {
    suspend fun retrieve(query: String, topK: Int): List<RetrievedChunk>
}
