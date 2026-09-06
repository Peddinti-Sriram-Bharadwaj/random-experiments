package com.example.nanoassistant.ui

import android.content.Context
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeech.QUEUE_FLUSH

object TtsHelper {
    private var tts: TextToSpeech? = null

    fun speak(context: Context, text: String) {
        if (tts == null) {
            tts = TextToSpeech(context.applicationContext) { }
        }
        tts?.speak(text, QUEUE_FLUSH, null, "assistant_utterance")
    }
}
