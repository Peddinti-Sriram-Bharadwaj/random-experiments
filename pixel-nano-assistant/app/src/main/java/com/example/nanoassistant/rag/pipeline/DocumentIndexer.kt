package com.example.nanoassistant.rag.pipeline

/** Stage 5: chunk and embed a document, inserting it into the store the retriever reads from. */
interface DocumentIndexer {
    suspend fun index(text: String, onChunkIndexed: (chunkIndex: Int) -> Unit = {})
}
