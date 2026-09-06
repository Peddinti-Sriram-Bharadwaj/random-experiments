package com.example.nanoassistant.rag.pipeline

/** Stage 1: turn the user's raw input into a better search query. */
interface QueryRewriter {
    suspend fun rewrite(query: String): String
}
