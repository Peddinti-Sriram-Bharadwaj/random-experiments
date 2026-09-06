package com.example.assistant

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.assistant.engine.LlmEngine
import com.example.assistant.engine.SttEngine
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class AssistantViewModel(
    private var stt: SttEngine,
    private val llm: LlmEngine,
    private val speak: (String) -> Unit,
    sttModelPath: String,
    llmModelPath: String
) : ViewModel() {

    private val _state = MutableStateFlow<AssistantState>(AssistantState.Idle)
    val state: StateFlow<AssistantState> = _state.asStateFlow()

    private val _sttEngineName = MutableStateFlow("Whisper base.en")
    val sttEngineName: StateFlow<String> = _sttEngineName.asStateFlow()

    private val history = mutableListOf<ConversationTurn>()

    init {
        viewModelScope.launch {
            try {
                stt.init(sttModelPath)
                llm.init(llmModelPath)
            } catch (e: Exception) {
                _state.value = AssistantState.Error("Model load failed: ${e.message}")
            }
        }
    }

    /** Hot-swaps the STT engine — releases the old native context before loading the new one. */
    fun switchSttEngine(name: String, newEngine: SttEngine, modelPath: String) {
        viewModelScope.launch {
            _state.value = AssistantState.SwitchingModel(name)
            stt.release()
            try {
                newEngine.init(modelPath)
                stt = newEngine
                _sttEngineName.value = name
                _state.value = AssistantState.Idle
            } catch (e: Exception) {
                _state.value = AssistantState.Error("Failed to load $name: ${e.message}")
            }
        }
    }

    fun onSpeechCaptured(pcm: FloatArray) {
        viewModelScope.launch {
            _state.value = AssistantState.Transcribing
            val text = try {
                stt.transcribe(pcm)
            } catch (e: Exception) {
                _state.value = AssistantState.Error("Transcription failed: ${e.message}")
                return@launch
            }

            if (text.isBlank()) {
                _state.value = AssistantState.Idle
                return@launch
            }

            history += ConversationTurn(Role.USER, text)
            _state.value = AssistantState.Thinking(partialResponse = "")

            val prompt = buildPrompt(history)
            val responseBuilder = StringBuilder()
            llm.generate(prompt).collect { token ->
                responseBuilder.append(token)
                _state.value = AssistantState.Thinking(responseBuilder.toString())
            }

            val fullResponse = responseBuilder.toString()
            history += ConversationTurn(Role.ASSISTANT, fullResponse)
            _state.value = AssistantState.Speaking(fullResponse)
            speak(fullResponse)
            _state.value = AssistantState.Idle
        }
    }

    fun onMicTapped() {
        _state.value = AssistantState.Listening
    }

    override fun onCleared() {
        stt.release()
        llm.release()
    }
}
