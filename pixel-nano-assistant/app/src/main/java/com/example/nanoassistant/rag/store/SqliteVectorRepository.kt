package com.example.nanoassistant.rag.store

import android.content.Context
import com.example.nanoassistant.rag.model.ChunkMetadata
import com.example.nanoassistant.rag.model.RetrievedChunk
import com.google.ai.edge.localagents.rag.memory.SqliteVectorStore
import com.google.ai.edge.localagents.rag.memory.VectorStoreRecord
import com.google.common.collect.ImmutableList
import java.io.File
import kotlin.math.sqrt

/**
 * Persistent on-disk vector store, backed by the AI Edge RAG SDK's native SqliteVectorStore
 * (a real SQLite database under the hood, JNI-backed — already bundled in the SDK we depend
 * on, so no extra native dependency). Unlike [InMemoryVectorRepository], the index survives
 * app restarts.
 *
 * Uses the SDK's default schema as-is (its custom-TableConfig constructor's native argument
 * order isn't documented, and guessing it wrong silently corrupts the schema) and tracks
 * ChunkMetadata in a side map keyed by chunk text — the same approach InMemoryVectorRepository
 * uses, since the SDK's own store has no query-by-key API to look it up otherwise.
 *
 * The store needs its embedding dimension at construction time, but callers only learn that
 * from the embedder at runtime — so the underlying store is created lazily on first insert.
 */
class SqliteVectorRepository(
    context: Context,
    dbFileName: String = "vectors.db"
) : VectorRepository {

    private val dbPath = File(context.filesDir, dbFileName).absolutePath
    private var store: SqliteVectorStore? = null
    private val metadataByText = mutableMapOf<String, ChunkMetadata>()

    override fun insert(text: String, embedding: List<Float>, metadata: ChunkMetadata) {
        val vectorStore = store ?: SqliteVectorStore(embedding.size, dbPath).also { store = it }
        vectorStore.insert(VectorStoreRecord.create(text, ImmutableList.copyOf(embedding)))
        metadataByText[text] = metadata
    }

    override fun findNearest(queryEmbedding: List<Float>, topK: Int): List<RetrievedChunk> {
        val vectorStore = store ?: return emptyList()
        return vectorStore.getNearestRecords(queryEmbedding, topK, 0.0f).map { record ->
            RetrievedChunk(record.data, cosineSimilarity(queryEmbedding, record.embeddings), metadataByText[record.data])
        }
    }

    private fun cosineSimilarity(a: List<Float>, b: List<Float>): Float {
        var dot = 0f; var normA = 0f; var normB = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        val denom = sqrt(normA) * sqrt(normB)
        return if (denom == 0f) 0f else dot / denom
    }
}
