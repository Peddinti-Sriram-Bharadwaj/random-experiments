package com.example.nanoassistant.rag.store

import com.example.nanoassistant.rag.model.ChunkMetadata
import com.example.nanoassistant.rag.model.RetrievedChunk

/** Lexical (keyword) search over indexed chunks — the counterpart to [VectorRepository]. */
interface KeywordIndex {
    fun insert(text: String, metadata: ChunkMetadata)
    fun search(query: String, topK: Int): List<RetrievedChunk>
}
