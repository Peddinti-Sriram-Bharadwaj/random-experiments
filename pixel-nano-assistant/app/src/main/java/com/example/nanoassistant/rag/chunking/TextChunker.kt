package com.example.nanoassistant.rag.chunking

interface TextChunker {
    fun chunk(text: String): List<String>
}
