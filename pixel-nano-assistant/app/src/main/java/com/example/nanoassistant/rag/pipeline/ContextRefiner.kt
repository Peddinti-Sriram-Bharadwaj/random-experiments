package com.example.nanoassistant.rag.pipeline

import com.example.nanoassistant.rag.model.RetrievedChunk

/** Stage 4: turn reranked chunks into the actual context string handed to the generator —
 *  dedup, trim, and format live here rather than being scattered through the pipeline. */
interface ContextRefiner {
    fun refine(chunks: List<RetrievedChunk>): String
}
