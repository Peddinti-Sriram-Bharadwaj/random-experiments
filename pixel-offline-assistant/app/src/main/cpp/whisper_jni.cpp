#include <jni.h>
#include <string>
#include <cmath>
#include <android/log.h>
#include "whisper.h"

#define LOG_TAG "whisper_jni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_assistant_engine_WhisperNative_nativeInit(JNIEnv* env, jobject /*thiz*/, jstring modelPath) {
    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    whisper_context_params cparams = whisper_context_default_params();
    // Vulkan encode path currently produces bad output on this device (whisper_full
    // succeeds but yields 0 segments) — force CPU until that's root-caused separately.
    cparams.use_gpu = false;
    whisper_context* ctx = whisper_init_from_file_with_params(path, cparams);
    env->ReleaseStringUTFChars(modelPath, path);
    return reinterpret_cast<jlong>(ctx);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_assistant_engine_WhisperNative_nativeTranscribe(
        JNIEnv* env, jobject /*thiz*/, jlong ctxPtr, jfloatArray pcm) {
    auto* ctx = reinterpret_cast<whisper_context*>(ctxPtr);
    if (!ctx) return env->NewStringUTF("");

    jsize n = env->GetArrayLength(pcm);
    LOGI("transcribe: n_samples=%d (%.1f sec of audio)", n, n / 16000.0f);
    jfloat* samples = env->GetFloatArrayElements(pcm, nullptr);

    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) if (std::abs(samples[i]) > max_abs) max_abs = std::abs(samples[i]);
    LOGI("transcribe: peak amplitude=%.5f", max_abs);

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.n_threads = 4;
    wparams.print_progress = false;
    wparams.print_special = false;
    // Tuned for our short push-to-talk clips (~2-5s) — whisper.cpp's default no-speech
    // heuristic is calibrated for long-form transcription and can wrongly zero out short audio.
    wparams.no_speech_thold = 0.1f;
    wparams.single_segment = true;

    LOGI("transcribe: params no_speech_thold=%.2f single_segment=%d language=%s",
         wparams.no_speech_thold, wparams.single_segment, wparams.language ? wparams.language : "null");
    LOGI("transcribe: calling whisper_full...");
    int result = whisper_full(ctx, wparams, samples, n);
    LOGI("transcribe: whisper_full returned %d", result);
    env->ReleaseFloatArrayElements(pcm, samples, JNI_ABORT);
    if (result != 0) return env->NewStringUTF("");

    std::string out;
    int n_segments = whisper_full_n_segments(ctx);
    LOGI("transcribe: n_segments=%d", n_segments);
    for (int i = 0; i < n_segments; i++) {
        out += whisper_full_get_segment_text(ctx, i);
    }
    return env->NewStringUTF(out.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_assistant_engine_WhisperNative_nativeFree(JNIEnv* /*env*/, jobject /*thiz*/, jlong ctxPtr) {
    auto* ctx = reinterpret_cast<whisper_context*>(ctxPtr);
    if (ctx) whisper_free(ctx);
}
