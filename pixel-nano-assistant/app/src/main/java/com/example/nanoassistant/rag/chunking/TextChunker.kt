package com.example.nanoassistant.rag.chunking

interface TextChunker {
    suspend fun chunk(text: String): List<String>
}
