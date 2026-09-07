package com.example.nanoassistant.rag.chunking

/** Splits on blank lines, then hard-wraps any paragraph still too long for one chunk. */
class ParagraphTextChunker(private val maxChars: Int = 500) : TextChunker {

    override suspend fun chunk(text: String): List<String> {
        val paragraphs = text.split(Regex("\n\\s*\n")).map { it.trim() }.filter { it.isNotEmpty() }
        val chunks = mutableListOf<String>()
        for (p in paragraphs) {
            if (p.length <= maxChars) {
                chunks += p
            } else {
                var start = 0
                while (start < p.length) {
                    val end = minOf(start + maxChars, p.length)
                    chunks += p.substring(start, end)
                    start = end
                }
            }
        }
        return chunks
    }
}
