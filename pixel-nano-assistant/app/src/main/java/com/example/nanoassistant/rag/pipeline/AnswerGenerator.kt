package com.example.nanoassistant.rag.pipeline

/** Stage 6: produce the final answer from the (rewritten) query and refined context. */
interface AnswerGenerator {
    suspend fun generate(query: String, context: String): String
}
