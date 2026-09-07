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
    // Merged chunks get embedded again afterward (by ChunkingDocumentIndexer, once per final
    // chunk) — that call has the same 256-token embedder limit as the per-sentence calls below,
    // and 1500 chars was far past it for this corpus's denser PDF text. 400 is comfortably under
    // the 500-char window SlidingWindowTextChunker already uses safely on the same corpus.
    private val maxChunkChars: Int = 400
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

    // The embedder has a fixed max input length (256 tokens for Gecko); a punctuation-free
    // stretch of text (a PDF table-of-contents block, a run of metadata labels with no
    // sentence-ending punctuation) can produce a "sentence" from the regex split below that
    // blows past that limit and crashes the native embedding call. A 700-char cap still hit
    // 346 tokens once (~2 chars/token — dense numeric/symbol-heavy PDF fragments tokenize much
    // less efficiently than prose), so this is deliberately far under any plausible ratio.
    private val maxSentenceChars = 300

    private fun splitIntoSentences(text: String): List<String> =
        text.trim()
            .split(Regex("(?<=[.!?])\\s+"))
            .map { it.trim() }
            .filter { it.isNotEmpty() }
            .flatMap { sentence -> sentence.chunked(maxSentenceChars) }

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
