package com.example.nanoassistant.rag.pipeline

import com.google.mlkit.genai.prompt.GenerativeModel

/** Asks Gemini Nano to turn a possibly-terse user query into a clearer standalone search
 *  query, which tends to improve embedding-based retrieval. Falls back to the raw query
 *  on any failure — a rewrite stage should never be the reason a question can't be answered. */
class NanoQueryRewriter(private val model: GenerativeModel) : QueryRewriter {

    override suspend fun rewrite(query: String): String = try {
        val prompt = "Rewrite the following as a single, clear, standalone search query. " +
            "Reply with only the rewritten query, no explanation.\n\nQuery: $query"
        val response = model.generateContent(prompt)
        response.candidates.firstOrNull()?.text?.trim()?.takeIf { it.isNotBlank() } ?: query
    } catch (e: Exception) {
        query
    }
}
