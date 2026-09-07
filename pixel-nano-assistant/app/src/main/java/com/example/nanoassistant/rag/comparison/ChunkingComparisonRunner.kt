package com.example.nanoassistant.rag.comparison

import com.example.nanoassistant.rag.chunking.SemanticTextChunker
import com.example.nanoassistant.rag.chunking.SlidingWindowTextChunker
import com.example.nanoassistant.rag.chunking.TextChunker
import com.example.nanoassistant.rag.embedding.EmbeddingService
import com.example.nanoassistant.rag.model.RetrievedChunk
import com.example.nanoassistant.rag.pipeline.ChunkingDocumentIndexer
import com.example.nanoassistant.rag.pipeline.CompositeRetriever
import com.example.nanoassistant.rag.pipeline.EmbeddingRetriever
import com.example.nanoassistant.rag.pipeline.KeywordRetriever
import com.example.nanoassistant.rag.pipeline.LexicalOverlapReranker
import com.example.nanoassistant.rag.store.Bm25KeywordIndex
import com.example.nanoassistant.rag.store.InMemoryVectorRepository
import com.example.nanoassistant.rag.store.KeywordIndex
import com.example.nanoassistant.rag.store.VectorRepository

/**
 * Indexes the same corpus twice — once per chunking strategy — into disposable in-memory
 * stores, then runs the same fixed query set through each and reports the top-k results
 * side by side. There's no labeled relevance ground truth here, so this is a qualitative
 * side-by-side (read the retrieved chunks yourself), not a precision/recall score.
 */
class ChunkingComparisonRunner(private val embeddingService: EmbeddingService) {

    data class ChunkerResult(val chunkCount: Int, val topChunks: List<RetrievedChunk>)
    data class QueryComparison(
        val query: String,
        val slidingWindow: ChunkerResult,
        val semantic: ChunkerResult
    )

    private class IndexedCorpus(val vectorRepository: VectorRepository, val keywordIndex: KeywordIndex, val chunkCount: Int)

    suspend fun compare(docs: Map<String, String>, queries: List<String>, topK: Int = 3): List<QueryComparison> {
        val slidingCorpus = index(SlidingWindowTextChunker(), docs)
        val semanticCorpus = index(SemanticTextChunker(embeddingService), docs)

        return queries.map { query ->
            QueryComparison(
                query = query,
                slidingWindow = ChunkerResult(slidingCorpus.chunkCount, retrieveTop(slidingCorpus, query, topK)),
                semantic = ChunkerResult(semanticCorpus.chunkCount, retrieveTop(semanticCorpus, query, topK))
            )
        }
    }

    private suspend fun index(chunker: TextChunker, docs: Map<String, String>): IndexedCorpus {
        val vectorRepository = InMemoryVectorRepository()
        val keywordIndex = Bm25KeywordIndex()
        val indexer = ChunkingDocumentIndexer(chunker, embeddingService, vectorRepository, keywordIndex)
        var totalChunks = 0
        for ((sourceId, text) in docs) {
            var docChunks = 0
            indexer.index(text, sourceId) { indexed -> docChunks = indexed }
            totalChunks += docChunks
        }
        return IndexedCorpus(vectorRepository, keywordIndex, totalChunks)
    }

    private suspend fun retrieveTop(corpus: IndexedCorpus, query: String, topK: Int): List<RetrievedChunk> {
        val retriever = CompositeRetriever(
            listOf(
                EmbeddingRetriever(embeddingService, corpus.vectorRepository),
                KeywordRetriever(corpus.keywordIndex)
            )
        )
        val retrieved = retriever.retrieve(query, topK * 2)
        return LexicalOverlapReranker().rerank(query, retrieved).take(topK)
    }
}
