package com.example.nanoassistant.rag.pipeline

import com.google.mlkit.genai.prompt.GenerativeModel

class NanoAnswerGenerator(private val model: GenerativeModel) : AnswerGenerator {

    override suspend fun generate(query: String, context: String): String {
        val prompt = """
            Context:
            $context

            Question: $query
            Answer using only the context above. If the context doesn't contain the answer, say so.
        """.trimIndent()
        val response = model.generateContent(prompt)
        return response.candidates.firstOrNull()?.text.orEmpty()
    }
}
