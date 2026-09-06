package com.example.nanoassistant.rag.pipeline

import com.example.nanoassistant.rag.model.RetrievedChunk

/** Hybrid rerank: blends the vector-similarity score with plain keyword overlap, since embedding
 *  similarity alone can miss exact-term matches (e.g. model names, numbers) that matter a lot
 *  for factual Q&A. A simple, dependency-free complement to semantic search. */
class LexicalOverlapReranker(private val lexicalWeight: Float = 0.3f) : Reranker {

    override fun rerank(query: String, chunks: List<RetrievedChunk>): List<RetrievedChunk> {
        val queryTerms = tokenize(query)
        if (queryTerms.isEmpty()) return chunks

        return chunks
            .map { chunk ->
                val chunkTerms = tokenize(chunk.text)
                val overlap = queryTerms.intersect(chunkTerms).size.toFloat() / queryTerms.size
                val blended = (1 - lexicalWeight) * chunk.score + lexicalWeight * overlap
                chunk.copy(score = blended)
            }
            .sortedByDescending { it.score }
    }

    private fun tokenize(text: String): Set<String> =
        text.lowercase().split(Regex("\\W+")).filter { it.length > 2 }.toSet()
}
