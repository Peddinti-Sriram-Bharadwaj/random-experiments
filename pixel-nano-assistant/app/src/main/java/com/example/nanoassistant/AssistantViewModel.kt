package com.example.nanoassistant

import com.google.mlkit.genai.common.FeatureStatus
import com.google.mlkit.genai.prompt.Generation
import com.google.mlkit.genai.prompt.GenerativeModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class AssistantViewModel(private val speak: (String) -> Unit) : ViewModel() {

    private val _state = MutableStateFlow<AssistantState>(AssistantState.Idle)
    val state: StateFlow<AssistantState> = _state.asStateFlow()

    private val generativeModel: GenerativeModel = Generation.getClient()

    init {
        viewModelScope.launch {
            try {
                when (generativeModel.checkStatus()) {
                    FeatureStatus.DOWNLOADABLE, FeatureStatus.DOWNLOADING -> {
                        _state.value = AssistantState.Transcribing // reused as "preparing model" indicator
                        generativeModel.download().collect { }
                    }
                    else -> Unit
                }
                _state.value = AssistantState.Idle
            } catch (e: Exception) {
                _state.value = AssistantState.Error("Gemini Nano unavailable: ${e.message}")
            }
        }
    }

    fun onMicTapped() {
        _state.value = AssistantState.Listening
    }

    fun onTranscript(text: String) {
        if (text.isBlank()) {
            _state.value = AssistantState.Idle
            return
        }
        viewModelScope.launch {
            _state.value = AssistantState.Thinking("")
            val startMs = System.currentTimeMillis()
            try {
                val response = generativeModel.generateContent(text)
                val elapsedMs = System.currentTimeMillis() - startMs
                val text = response.candidates.firstOrNull()?.text.orEmpty()
                _state.value = AssistantState.Speaking(text, elapsedMs)
                speak(text)
                _state.value = AssistantState.Idle
            } catch (e: Exception) {
                _state.value = AssistantState.Error("Generation failed: ${e.message}")
            }
        }
    }

    fun onTranscriptError(message: String) {
        _state.value = AssistantState.Error(message)
    }
}
