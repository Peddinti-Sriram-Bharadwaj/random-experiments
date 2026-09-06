package com.example.nanoassistant.rag.pipeline

import com.example.nanoassistant.rag.chunking.TextChunker
import com.example.nanoassistant.rag.embedding.EmbeddingService
import com.example.nanoassistant.rag.store.VectorRepository

class ChunkingDocumentIndexer(
    private val chunker: TextChunker,
    private val embeddingService: EmbeddingService,
    private val vectorRepository: VectorRepository
) : DocumentIndexer {

    override suspend fun index(text: String, onChunkIndexed: (Int) -> Unit) {
        val chunks = chunker.chunk(text)
        chunks.forEachIndexed { index, chunk ->
            val embedding = embeddingService.embedDocument(chunk)
            vectorRepository.insert(chunk, embedding)
            onChunkIndexed(index + 1)
        }
    }
}
