package com.example.nanoassistant.rag.store

import android.content.Context
import com.example.nanoassistant.rag.model.ChunkMetadata
import org.json.JSONObject
import java.io.File

/**
 * Durable record of every (text, metadata) pair ever indexed — one JSON object per line.
 *
 * The vector store (SqliteVectorStore) is already persistent on its own, but the BM25 keyword
 * index only ever lives in memory (it rescores its whole corpus at query time, so there's
 * nothing else to persist it against). This manifest is what lets that in-memory index be
 * rebuilt on the next launch without re-running extraction/chunking/embedding — reading it back
 * is just string parsing, no embedder calls.
 */
class PersistedChunkStore(context: Context, fileName: String = "chunk_manifest.jsonl") {

    private val file = File(context.filesDir, fileName)

    fun exists(): Boolean = file.exists() && file.length() > 0L

    fun append(text: String, metadata: ChunkMetadata) {
        val json = JSONObject().apply {
            put("text", text)
            put("sourceId", metadata.sourceId)
            put("chunkIndex", metadata.chunkIndex)
            metadata.section?.let { put("section", it) }
        }
        file.appendText(json.toString() + "\n")
    }

    fun readAll(): List<Pair<String, ChunkMetadata>> {
        if (!file.exists()) return emptyList()
        return file.readLines()
            .filter { it.isNotBlank() }
            .map { line ->
                val json = JSONObject(line)
                val metadata = ChunkMetadata(
                    sourceId = json.getString("sourceId"),
                    chunkIndex = json.getInt("chunkIndex"),
                    section = if (json.has("section")) json.getString("section") else null
                )
                json.getString("text") to metadata
            }
    }

    fun clear() {
        file.delete()
    }
}
