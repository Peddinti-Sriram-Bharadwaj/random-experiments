package com.example.nanoassistant.rag.pipeline

import com.example.nanoassistant.rag.chunking.TextChunker
import com.example.nanoassistant.rag.embedding.EmbeddingService
import com.example.nanoassistant.rag.model.ChunkMetadata
import com.example.nanoassistant.rag.store.KeywordIndex
import com.example.nanoassistant.rag.store.VectorRepository

/** Dual-writes each chunk into both the vector store and the keyword index — the standard
 *  shape for a hybrid-search insert path, since embedding and lexical retrieval each need
 *  their own representation of the same text. */
class ChunkingDocumentIndexer(
    private val chunker: TextChunker,
    private val embeddingService: EmbeddingService,
    private val vectorRepository: VectorRepository,
    private val keywordIndex: KeywordIndex
) : DocumentIndexer {

    override suspend fun index(text: String, sourceId: String, onChunkIndexed: (Int) -> Unit) {
        val chunks = chunker.chunk(text)
        chunks.forEachIndexed { index, chunk ->
            val metadata = ChunkMetadata(sourceId = sourceId, chunkIndex = index)
            val embedding = embeddingService.embedDocument(chunk)
            vectorRepository.insert(chunk, embedding, metadata)
            keywordIndex.insert(chunk, metadata)
            onChunkIndexed(index + 1)
        }
    }
}
