package com.example.nanoassistant.rag.chunking

import com.example.nanoassistant.rag.embedding.EmbeddingService
import kotlin.math.sqrt

/**
 * Splits at sentence boundaries where meaning actually shifts, instead of at a fixed length:
 * embed every sentence, walk through them in order, and start a new chunk whenever cosine
 * similarity to the previous sentence drops below [breakpointThreshold]. Costs one embedding
 * call per sentence at index time (more expensive than [SlidingWindowTextChunker]) in exchange
 * for chunks that stay topically coherent instead of just length-bounded.
 */
class SemanticTextChunker(
    private val embeddingService: EmbeddingService,
    private val breakpointThreshold: Double = 0.5,
    private val maxChunkChars: Int = 1500
) : TextChunker {

    override suspend fun chunk(text: String): List<String> {
        val sentences = splitIntoSentences(text)
        if (sentences.isEmpty()) return emptyList()
        if (sentences.size == 1) return sentences

        val embeddings = sentences.map { embeddingService.embedDocument(it) }

        val chunks = mutableListOf<String>()
        val current = StringBuilder(sentences[0])
        for (i in 1 until sentences.size) {
            val similarity = cosineSimilarity(embeddings[i - 1], embeddings[i])
            val wouldOverflow = current.length + sentences[i].length + 1 > maxChunkChars
            if (similarity < breakpointThreshold || wouldOverflow) {
                chunks += current.toString().trim()
                current.setLength(0)
                current.append(sentences[i])
            } else {
                current.append(' ').append(sentences[i])
            }
        }
        if (current.isNotEmpty()) chunks += current.toString().trim()
        return chunks
    }

    private fun splitIntoSentences(text: String): List<String> =
        text.trim()
            .split(Regex("(?<=[.!?])\\s+"))
            .map { it.trim() }
            .filter { it.isNotEmpty() }

    private fun cosineSimilarity(a: List<Float>, b: List<Float>): Double {
        var dot = 0.0; var normA = 0.0; var normB = 0.0
        for (i in a.indices) {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        val denom = sqrt(normA) * sqrt(normB)
        return if (denom == 0.0) 0.0 else dot / denom
    }
}
