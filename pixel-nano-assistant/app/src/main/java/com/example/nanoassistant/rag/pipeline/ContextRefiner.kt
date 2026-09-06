package com.example.nanoassistant.rag.pipeline

import com.example.nanoassistant.rag.model.RetrievedChunk

/** Stage 4: turn reranked chunks into the actual context string handed to the generator —
 *  dedup, trim, filter, and format live here rather than being scattered through the pipeline.
 *  Takes the query too, since deciding what's relevant (e.g. System 2 Attention-style context
 *  filtering) requires knowing what's being asked, not just what was retrieved. */
interface ContextRefiner {
    suspend fun refine(query: String, chunks: List<RetrievedChunk>): String
}
