package com.example.nanoassistant.rag.model

/** Provenance for one chunk: which document it came from and where within it. */
data class ChunkMetadata(
    val sourceId: String,
    val chunkIndex: Int,
    val section: String? = null
)
