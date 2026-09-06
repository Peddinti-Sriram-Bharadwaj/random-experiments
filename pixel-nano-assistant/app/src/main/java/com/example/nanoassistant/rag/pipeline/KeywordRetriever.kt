package com.example.nanoassistant.rag.pipeline

import com.example.nanoassistant.rag.model.RetrievedChunk
import com.example.nanoassistant.rag.store.KeywordIndex

class KeywordRetriever(private val keywordIndex: KeywordIndex) : Retriever {
    override suspend fun retrieve(query: String, topK: Int): List<RetrievedChunk> =
        keywordIndex.search(query, topK)
}
