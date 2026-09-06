package com.example.nanoassistant.rag.pipeline

import com.example.nanoassistant.rag.model.RetrievedChunk

/** Stage 3: reorder (and optionally drop) retrieved candidates before they're used as context. */
interface Reranker {
    fun rerank(query: String, chunks: List<RetrievedChunk>): List<RetrievedChunk>
}
