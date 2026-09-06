package com.example.assistant.engine

interface SttEngine {
    suspend fun init(modelPath: String)
    suspend fun transcribe(pcm: FloatArray): String
    fun release()
}
