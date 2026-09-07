package com.example.nanoassistant.rag.store

import com.example.nanoassistant.rag.model.ChunkMetadata
import com.example.nanoassistant.rag.model.RetrievedChunk

/** Wraps a [KeywordIndex], additionally durably recording every insert to [chunkStore] so the
 *  index can be rebuilt on the next launch without re-running extraction/chunking/embedding. */
class PersistingKeywordIndex(
    private val delegate: KeywordIndex,
    private val chunkStore: PersistedChunkStore
) : KeywordIndex {

    override fun insert(text: String, metadata: ChunkMetadata) {
        delegate.insert(text, metadata)
        chunkStore.append(text, metadata)
    }

    override fun search(query: String, topK: Int): List<RetrievedChunk> = delegate.search(query, topK)
}
