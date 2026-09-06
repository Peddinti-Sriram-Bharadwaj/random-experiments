package com.example.nanoassistant

sealed class AssistantState {
    data object Idle : AssistantState()
    data object Listening : AssistantState()
    data object Transcribing : AssistantState()
    data class Thinking(val partialResponse: String) : AssistantState()
    data class Speaking(val fullResponse: String, val timingMs: Long) : AssistantState()
    data class Error(val message: String) : AssistantState()
}
