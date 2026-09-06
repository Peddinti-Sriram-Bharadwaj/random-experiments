#include <jni.h>
#include <string>
#include <vector>
#include <cstdlib>
#include <android/log.h>
#include "llama.h"

#define LOG_TAG "llama_jni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

// One-time backend init — llama.cpp requires this before any model load.
struct BackendGuard {
    BackendGuard() {
        // Cooperative-matrix / integer-dot-product Vulkan extensions are a known source of
        // silently-wrong compute results on mobile Mali/Adreno drivers (upstream llama.cpp
        // issues report exactly our symptom: quantized-model garbage output on Android GPUs).
        // Disable them and see if the encoder/decoder math becomes correct on GPU.
        setenv("GGML_VK_DISABLE_COOPMAT", "1", 1);
        setenv("GGML_VK_DISABLE_COOPMAT2", "1", 1);
        setenv("GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR", "1", 1);
        setenv("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT", "1", 1);
        llama_backend_init();
    }
    ~BackendGuard() { llama_backend_free(); }
};
static BackendGuard g_backend;

struct LlamaContext {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;
    llama_sampler* sampler = nullptr;
};

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_assistant_engine_LlamaNative_nativeInit(JNIEnv* env, jobject /*thiz*/, jstring modelPath) {
    const char* path = env->GetStringUTFChars(modelPath, nullptr);

    llama_model_params model_params = llama_model_default_params();
    // Testing GPU offload again now that COOPMAT/integer-dot-product are disabled above.
    // model_params.n_gpu_layers left at its default (-1 = offload everything).
    llama_model* model = llama_model_load_from_file(path, model_params);
    env->ReleaseStringUTFChars(modelPath, path);
    if (!model) return 0;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;      // context window — small on purpose for a phone
    ctx_params.n_batch = 512;

    llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        llama_model_free(model);
        return 0;
    }

    // Greedy sampler chain — simplest possible policy, always picks the highest-probability
    // token. Good enough to prove the pipeline works; swap for temp/top-p once it's running.
    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    auto* holder = new LlamaContext{model, ctx, llama_model_get_vocab(model), sampler};
    return reinterpret_cast<jlong>(holder);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_assistant_engine_LlamaNative_nativeGenerate(
        JNIEnv* env, jobject /*thiz*/, jlong ctxPtr, jstring promptJ, jobject callback) {
    auto* holder = reinterpret_cast<LlamaContext*>(ctxPtr);
    if (!holder) return;

    const char* promptChars = env->GetStringUTFChars(promptJ, nullptr);
    std::string prompt(promptChars);
    env->ReleaseStringUTFChars(promptJ, promptChars);

    jclass callbackClass = env->GetObjectClass(callback);
    jmethodID onToken = env->GetMethodID(callbackClass, "onToken", "(Ljava/lang/String;)Z");

    // Tokenize prompt
    const int n_prompt = -llama_tokenize(holder->vocab, prompt.c_str(), (int32_t) prompt.size(),
                                          nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_prompt);
    llama_tokenize(holder->vocab, prompt.c_str(), (int32_t) prompt.size(),
                   tokens.data(), n_prompt, true, true);

    LOGI("generate: n_prompt=%d first_ids=[%d,%d,%d,%d,%d] last_ids=[%d,%d,%d]",
         n_prompt,
         n_prompt > 0 ? tokens[0] : -1,
         n_prompt > 1 ? tokens[1] : -1,
         n_prompt > 2 ? tokens[2] : -1,
         n_prompt > 3 ? tokens[3] : -1,
         n_prompt > 4 ? tokens[4] : -1,
         n_prompt > 2 ? tokens[n_prompt-3] : -1,
         n_prompt > 1 ? tokens[n_prompt-2] : -1,
         n_prompt > 0 ? tokens[n_prompt-1] : -1);

    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());

    const int max_new_tokens = 256;
    for (int i = 0; i < max_new_tokens; i++) {
        int decode_result = llama_decode(holder->ctx, batch);
        if (i < 5) LOGI("generate: step %d decode_result=%d", i, decode_result);
        if (decode_result != 0) break;

        llama_token new_token = llama_sampler_sample(holder->sampler, holder->ctx, -1);
        if (i < 10) LOGI("generate: step %d sampled token_id=%d is_eog=%d", i, new_token, llama_vocab_is_eog(holder->vocab, new_token));
        if (llama_vocab_is_eog(holder->vocab, new_token)) break;

        char buf[256];
        int n = llama_token_to_piece(holder->vocab, new_token, buf, sizeof(buf), 0, true);
        jstring tokenStr = env->NewStringUTF(std::string(buf, n).c_str());
        jboolean cont = env->CallBooleanMethod(callback, onToken, tokenStr);
        env->DeleteLocalRef(tokenStr);
        if (!cont) break;

        batch = llama_batch_get_one(&new_token, 1);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_assistant_engine_LlamaNative_nativeFree(JNIEnv* /*env*/, jobject /*thiz*/, jlong ctxPtr) {
    auto* holder = reinterpret_cast<LlamaContext*>(ctxPtr);
    if (!holder) return;
    llama_sampler_free(holder->sampler);
    llama_free(holder->ctx);
    llama_model_free(holder->model);
    delete holder;
}
