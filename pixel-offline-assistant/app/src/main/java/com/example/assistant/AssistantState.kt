package com.example.assistant

sealed class AssistantState {
    data object Idle : AssistantState()
    data object Listening : AssistantState()
    data object Transcribing : AssistantState()
    data class Thinking(val partialResponse: String) : AssistantState()
    data class Speaking(val fullResponse: String) : AssistantState()
    data class SwitchingModel(val name: String) : AssistantState()
    data class Error(val message: String) : AssistantState()
}
