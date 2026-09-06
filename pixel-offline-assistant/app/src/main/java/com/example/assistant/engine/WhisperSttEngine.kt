package com.example.assistant.engine

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class WhisperSttEngine(private val native: WhisperNative = WhisperNative()) : SttEngine {

    private var ctxPtr: Long = 0L

    override suspend fun init(modelPath: String) = withContext(Dispatchers.Default) {
        ctxPtr = native.nativeInit(modelPath)
        check(ctxPtr != 0L) { "Failed to load Whisper model at $modelPath" }
    }

    override suspend fun transcribe(pcm: FloatArray): String = withContext(Dispatchers.Default) {
        native.nativeTranscribe(ctxPtr, pcm)
    }

    override fun release() {
        if (ctxPtr != 0L) {
            native.nativeFree(ctxPtr)
            ctxPtr = 0L
        }
    }
}
