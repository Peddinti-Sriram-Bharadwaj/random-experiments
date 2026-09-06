package com.example.nanoassistant.rag.embedding

import com.google.ai.edge.localagents.rag.models.EmbedData
import com.google.ai.edge.localagents.rag.models.EmbeddingRequest
import com.google.ai.edge.localagents.rag.models.GeckoEmbeddingModel
import kotlinx.coroutines.guava.await
import java.util.Optional

/** Adapts Google's Gecko (Gemma-family) on-device embedder to our [EmbeddingService] port. */
class GeckoEmbeddingService(modelPath: String, tokenizerPath: String) : EmbeddingService {

    private val embedder = GeckoEmbeddingModel(modelPath, Optional.of(tokenizerPath), false)

    override suspend fun embedDocument(text: String): List<Float> =
        embed(text, EmbedData.TaskType.RETRIEVAL_DOCUMENT)

    override suspend fun embedQuery(text: String): List<Float> =
        embed(text, EmbedData.TaskType.RETRIEVAL_QUERY)

    private suspend fun embed(text: String, taskType: EmbedData.TaskType): List<Float> {
        val request = EmbeddingRequest.create(listOf(EmbedData.create(text, taskType)))
        return embedder.getEmbeddings(request).await()
    }
}
