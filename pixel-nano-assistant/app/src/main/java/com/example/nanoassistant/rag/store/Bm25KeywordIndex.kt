package com.example.nanoassistant.rag.store

import com.example.nanoassistant.rag.model.RetrievedChunk
import kotlin.math.ln

/**
 * Classic BM25 (Robertson/Sparck-Jones) over an in-memory corpus, rescored fully at query time —
 * fine at the chunk counts a phone-local RAG corpus actually has; no incremental index needed.
 *
 * Two things naive term-overlap scoring gets wrong that BM25 fixes:
 *  - term-frequency saturation: matching a word 10 times isn't 10x as relevant as matching once
 *  - inverse document frequency: a rare term matching is worth more than a common one matching
 */
class Bm25KeywordIndex(private val k1: Double = 1.5, private val b: Double = 0.75) : KeywordIndex {

    private val documents = mutableListOf<List<String>>()
    private val rawText = mutableListOf<String>()

    override fun insert(text: String) {
        rawText += text
        documents += tokenize(text)
    }

    override fun search(query: String, topK: Int): List<RetrievedChunk> {
        if (documents.isEmpty()) return emptyList()

        val queryTerms = tokenize(query).toSet()
        val avgDocLength = documents.map { it.size }.average()
        val docCount = documents.size

        val docFrequency = queryTerms.associateWith { term ->
            documents.count { doc -> term in doc }
        }

        return documents.indices
            .map { i -> RetrievedChunk(rawText[i], bm25Score(documents[i], queryTerms, docFrequency, docCount, avgDocLength)) }
            .filter { it.score > 0f }
            .sortedByDescending { it.score }
            .take(topK)
    }

    private fun bm25Score(
        doc: List<String>,
        queryTerms: Set<String>,
        docFrequency: Map<String, Int>,
        docCount: Int,
        avgDocLength: Double
    ): Float {
        val termFrequency = doc.groupingBy { it }.eachCount()
        var score = 0.0
        for (term in queryTerms) {
            val df = docFrequency[term] ?: 0
            if (df == 0) continue
            val idf = ln((docCount - df + 0.5) / (df + 0.5) + 1)
            val tf = termFrequency[term] ?: 0
            val numerator = tf * (k1 + 1)
            val denominator = tf + k1 * (1 - b + b * (doc.size / avgDocLength))
            score += idf * (numerator / denominator)
        }
        return score.toFloat()
    }

    private fun tokenize(text: String): List<String> =
        text.lowercase().split(Regex("\\W+")).filter { it.length > 2 }
}
