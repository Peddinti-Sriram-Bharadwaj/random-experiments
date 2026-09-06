package com.example.assistant.engine

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext

class LlamaLlmEngine(private val native: LlamaNative = LlamaNative()) : LlmEngine {

    private var ctxPtr: Long = 0L

    override suspend fun init(modelPath: String) = withContext(Dispatchers.Default) {
        ctxPtr = native.nativeInit(modelPath)
        check(ctxPtr != 0L) { "Failed to load LLM model at $modelPath" }
    }

    override fun generate(prompt: String): Flow<String> = callbackFlow {
        val callback = TokenCallback { token ->
            val sent = trySend(token).isSuccess
            sent // returning false from onToken stops native generation loop
        }
        native.nativeGenerate(ctxPtr, prompt, callback)
        close()
        awaitClose { /* nothing to cancel mid-native-call in this simple version */ }
    }.flowOn(Dispatchers.Default)

    override fun release() {
        if (ctxPtr != 0L) {
            native.nativeFree(ctxPtr)
            ctxPtr = 0L
        }
    }
}
