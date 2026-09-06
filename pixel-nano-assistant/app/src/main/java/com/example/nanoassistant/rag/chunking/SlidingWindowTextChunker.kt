package com.example.nanoassistant.rag.chunking

/**
 * Fixed-size chunks with overlap: each window starts [overlapChars] before the previous one
 * ended, so a sentence sitting on a boundary still appears whole in at least one chunk — the
 * thing [ParagraphTextChunker]'s hard paragraph cuts lose.
 */
class SlidingWindowTextChunker(
    private val windowChars: Int = 500,
    private val overlapChars: Int = 100
) : TextChunker {

    init {
        require(overlapChars < windowChars) { "overlap must be smaller than the window" }
    }

    override suspend fun chunk(text: String): List<String> {
        val normalized = text.trim()
        if (normalized.isEmpty()) return emptyList()

        val chunks = mutableListOf<String>()
        val stride = windowChars - overlapChars
        var start = 0
        while (start < normalized.length) {
            val end = minOf(start + windowChars, normalized.length)
            chunks += normalized.substring(start, end).trim()
            if (end == normalized.length) break
            start += stride
        }
        return chunks.filter { it.isNotEmpty() }
    }
}
