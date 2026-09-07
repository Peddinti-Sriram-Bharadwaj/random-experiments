package com.example.nanoassistant.rag.embedding

import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * LRU-caches [embedQuery] results by normalized query text. Repeated or near-identical questions
 * are common on a phone (users re-ask, retry after a typo fix, or the same query gets rewritten
 * to the same thing twice), and each cache hit skips a full on-device embedder pass — real
 * latency/battery savings for a device-constrained app. [embedDocument] is never cached: chunk
 * text is effectively unique per document, so caching it would only grow memory for no hit rate.
 */
class CachingEmbeddingService(
    private val delegate: EmbeddingService,
    private val maxEntries: Int = 64
) : EmbeddingService {

    private val mutex = Mutex()
    private val cache = object : LinkedHashMap<String, List<Float>>(maxEntries, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, List<Float>>) =
            size > maxEntries
    }

    override suspend fun embedDocument(text: String): List<Float> = delegate.embedDocument(text)

    override suspend fun embedQuery(text: String): List<Float> {
        val key = text.trim().lowercase()
        mutex.withLock { cache[key] }?.let { return it }

        val embedding = delegate.embedQuery(text)
        mutex.withLock { cache[key] = embedding }
        return embedding
    }
}
