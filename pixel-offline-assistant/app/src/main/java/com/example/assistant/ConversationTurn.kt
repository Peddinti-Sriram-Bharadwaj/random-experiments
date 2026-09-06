package com.example.assistant

enum class Role { USER, ASSISTANT }

data class ConversationTurn(
    val role: Role,
    val text: String,
    val timestampMs: Long = System.currentTimeMillis()
)

/** Qwen2.5 chat template. Pure function — testable with no native code involved. */
fun buildPrompt(history: List<ConversationTurn>): String = buildString {
    for (turn in history) {
        val roleTag = if (turn.role == Role.USER) "user" else "assistant"
        append("<|im_start|>$roleTag\n${turn.text}\n<|im_end|>\n")
    }
    append("<|im_start|>assistant\n")
}
