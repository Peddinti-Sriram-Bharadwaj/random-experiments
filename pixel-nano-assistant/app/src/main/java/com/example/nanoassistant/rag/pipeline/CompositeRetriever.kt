package com.example.nanoassistant.rag.pipeline

import com.example.nanoassistant.rag.model.RetrievedChunk

/**
 * Fuses multiple retrievers' ranked lists via Reciprocal Rank Fusion, so keyword (BM25) and
 * embedding retrieval each get a real vote in which candidates survive — not just a rerank of
 * whatever embedding retrieval alone happened to surface. RRF combines *rankings*, not raw
 * scores, which sidesteps the problem that BM25 scores and cosine similarities aren't on
 * comparable scales.
 *
 * score(chunk) = sum over lists containing it of 1 / (rrfK + rank_in_that_list)
 */
class CompositeRetriever(
    private val retrievers: List<Retriever>,
    private val poolMultiplier: Int = 3,
    private val rrfK: Int = 60
) : Retriever {

    override suspend fun retrieve(query: String, topK: Int): List<RetrievedChunk> {
        val pool = maxOf(topK * poolMultiplier, 10)
        val rankedLists = retrievers.map { it.retrieve(query, pool) }

        val fusedScores = LinkedHashMap<String, Float>()
        val chunkByText = LinkedHashMap<String, RetrievedChunk>()
        for (list in rankedLists) {
            list.forEachIndexed { rank, chunk ->
                fusedScores[chunk.text] = (fusedScores[chunk.text] ?: 0f) + 1f / (rrfK + rank + 1)
                // First list to surface a chunk wins its metadata — every list should agree on it
                // anyway since it's keyed by chunk text, not list-specific.
                chunkByText.putIfAbsent(chunk.text, chunk)
            }
        }

        return fusedScores.entries
            .sortedByDescending { it.value }
            .take(topK)
            .map { (text, score) -> chunkByText.getValue(text).copy(score = score) }
    }
}
