package com.example.assistant.ui

import android.content.Context
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeech.QUEUE_FLUSH

/**
 * Android's built-in TTS engine is already fully on-device — no model to manage here,
 * which is exactly why we're deferring the "neural TTS" upgrade until the base loop works.
 */
object TtsHelper {
    private var tts: TextToSpeech? = null

    fun speak(context: Context, text: String) {
        if (tts == null) {
            tts = TextToSpeech(context.applicationContext) { }
        }
        tts?.speak(text, QUEUE_FLUSH, null, "assistant_utterance")
    }
}
