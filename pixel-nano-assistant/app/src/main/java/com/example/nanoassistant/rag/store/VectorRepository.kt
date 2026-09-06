package com.example.nanoassistant.rag.store

import com.example.nanoassistant.rag.model.RetrievedChunk

/** Storage + similarity search for embedded text chunks. */
interface VectorRepository {
    fun insert(text: String, embedding: List<Float>)
    fun findNearest(queryEmbedding: List<Float>, topK: Int): List<RetrievedChunk>
}
