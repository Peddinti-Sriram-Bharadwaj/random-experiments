package com.example.assistant.engine

/** Thin JNI wrapper — mirrors LlamaNative's shape for consistency. */
class WhisperNative {
    external fun nativeInit(modelPath: String): Long
    external fun nativeTranscribe(ctxPtr: Long, pcm: FloatArray): String
    external fun nativeFree(ctxPtr: Long)

    companion object {
        init { System.loadLibrary("whisper_jni") }
    }
}
