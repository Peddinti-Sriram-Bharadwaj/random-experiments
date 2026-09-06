package com.example.nanoassistant.rag.pipeline

/** Null Object: does nothing. Useful as a fast default or for testing the rest of the pipeline
 *  without depending on Nano being available. */
class PassthroughQueryRewriter : QueryRewriter {
    override suspend fun rewrite(query: String): String = query
}
