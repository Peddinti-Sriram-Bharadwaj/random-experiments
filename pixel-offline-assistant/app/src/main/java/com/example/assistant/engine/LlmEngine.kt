package com.example.assistant.engine

import kotlinx.coroutines.flow.Flow

interface LlmEngine {
    suspend fun init(modelPath: String)
    fun generate(prompt: String): Flow<String>
    fun release()
}
