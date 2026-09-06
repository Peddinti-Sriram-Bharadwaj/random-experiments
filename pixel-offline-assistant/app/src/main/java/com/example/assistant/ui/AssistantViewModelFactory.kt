package com.example.assistant.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.example.assistant.AssistantViewModel
import com.example.assistant.engine.LlmEngine
import com.example.assistant.engine.SttEngine

class AssistantViewModelFactory(
    private val sttEngine: SttEngine,
    private val llmEngine: LlmEngine,
    private val speak: (String) -> Unit,
    private val sttModelPath: String,
    private val llmModelPath: String
) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        @Suppress("UNCHECKED_CAST")
        return AssistantViewModel(sttEngine, llmEngine, speak, sttModelPath, llmModelPath) as T
    }
}
