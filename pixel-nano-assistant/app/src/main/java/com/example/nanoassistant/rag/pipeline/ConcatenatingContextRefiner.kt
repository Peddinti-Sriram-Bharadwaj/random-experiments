package com.example.nanoassistant.rag.pipeline

import com.example.nanoassistant.rag.model.RetrievedChunk

/** Drops exact-duplicate chunks (common when overlapping paragraphs both match), then joins
 *  what's left up to a character budget so we don't blow past the generator's context window. */
class ConcatenatingContextRefiner(private val maxContextChars: Int = 2000) : ContextRefiner {

    override fun refine(chunks: List<RetrievedChunk>): String {
        val seen = LinkedHashSet<String>()
        val builder = StringBuilder()
        for (chunk in chunks) {
            if (!seen.add(chunk.text)) continue
            val addition = if (builder.isEmpty()) chunk.text else "\n\n${chunk.text}"
            if (builder.length + addition.length > maxContextChars) break
            builder.append(addition)
        }
        return builder.toString()
    }
}
