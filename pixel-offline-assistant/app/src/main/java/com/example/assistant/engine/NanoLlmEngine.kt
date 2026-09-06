package com.example.assistant.engine

import com.google.mlkit.genai.common.FeatureStatus
import com.google.mlkit.genai.prompt.Generation
import com.google.mlkit.genai.prompt.GenerativeModel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

/** Wraps Gemini Nano via AICore behind the same LlmEngine interface as our llama.cpp engine. */
class NanoLlmEngine : LlmEngine {

    private lateinit var model: GenerativeModel

    override suspend fun init(modelPath: String) {
        // modelPath is unused — Gemini Nano's model lives in AICore, not a file we manage.
        model = Generation.getClient()
        when (model.checkStatus()) {
            FeatureStatus.DOWNLOADABLE, FeatureStatus.DOWNLOADING -> model.download().collect { }
            else -> Unit
        }
    }

    override fun generate(prompt: String): Flow<String> = flow {
        val response = model.generateContent(prompt)
        emit(response.candidates.firstOrNull()?.text.orEmpty())
    }

    override fun release() = Unit
}
