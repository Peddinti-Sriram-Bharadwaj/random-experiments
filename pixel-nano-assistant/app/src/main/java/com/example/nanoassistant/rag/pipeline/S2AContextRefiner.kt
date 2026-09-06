package com.example.nanoassistant.rag.pipeline

import com.example.nanoassistant.rag.model.RetrievedChunk
import com.google.mlkit.genai.prompt.GenerativeModel

/**
 * System 2 Attention (Weston & Sukhbaatar, 2023): rather than handing the generator raw
 * retrieved chunks directly, first ask the model to *regenerate* the context, keeping only
 * the parts relevant to the query and dropping the rest. This is a second, separate LLM call
 * whose only job is context hygiene — it never sees the instruction "answer the question,"
 * only "extract what's relevant," which keeps it from being tempted to answer prematurely
 * or let irrelevant retrieved text bias the real generation step that follows.
 *
 * Falls back to [fallback] (plain concatenation) if the filtering call fails or returns
 * nothing usable — a context-hygiene step should never be the reason a query gets no answer.
 */
class S2AContextRefiner(
    private val model: GenerativeModel,
    private val fallback: ContextRefiner = ConcatenatingContextRefiner()
) : ContextRefiner {

    override suspend fun refine(query: String, chunks: List<RetrievedChunk>): String {
        val rawContext = fallback.refine(query, chunks)
        if (rawContext.isBlank()) return rawContext

        return try {
            val prompt = """
                Below is a set of retrieved text passages, and a question. Extract and output
                only the sentences or facts from the passages that are relevant to answering the
                question. Remove anything irrelevant, off-topic, or unrelated to the question.
                Do not answer the question — only output the filtered relevant text, verbatim
                from the passages, with no commentary.

                Passages:
                $rawContext

                Question: $query

                Filtered relevant text:
            """.trimIndent()

            val response = model.generateContent(prompt)
            val filtered = response.candidates.firstOrNull()?.text?.trim()
            filtered.takeUnless { it.isNullOrBlank() } ?: rawContext
        } catch (e: Exception) {
            rawContext
        }
    }
}
