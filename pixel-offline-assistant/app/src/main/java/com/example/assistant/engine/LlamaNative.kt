package com.example.assistant.engine

/** Thin JNI wrapper — no business logic lives here on purpose. */
class LlamaNative {
    external fun nativeInit(modelPath: String): Long
    external fun nativeGenerate(ctxPtr: Long, prompt: String, callback: TokenCallback)
    external fun nativeFree(ctxPtr: Long)

    companion object {
        init { System.loadLibrary("llama_jni") }
    }
}

fun interface TokenCallback {
    /** Return false to stop generation early (e.g. user cancelled). */
    fun onToken(token: String): Boolean
}
