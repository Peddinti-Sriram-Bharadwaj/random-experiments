package com.example.nanoassistant.rag.model

/** Everything a caller might want to show for one query — one field per pipeline stage's output. */
data class AskResult(
    val originalQuery: String,
    val rewrittenQuery: String,
    val retrievedChunks: List<RetrievedChunk>,
    val rerankedChunks: List<RetrievedChunk>,
    val refinedContext: String,
    val answer: String,
    val isLowConfidence: Boolean = false
)
