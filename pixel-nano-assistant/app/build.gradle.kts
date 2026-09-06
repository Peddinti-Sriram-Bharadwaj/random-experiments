plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.nanoassistant"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.nanoassistant"
        minSdk = 31          // GenAI Speech Recognition basic mode needs 31+; advanced mode is Pixel 10-only anyway
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        viewBinding = true
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.8.7")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("com.google.android.material:material:1.12.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")

    // Gemini Nano via AICore — no model files to manage, no JNI, no NDK.
    implementation("com.google.mlkit:genai-prompt:1.0.0-beta2")
    // STT uses Android's standard SpeechRecognizer (on-device mode) rather than ML Kit's
    // GenAI Speech Recognition API — that one is alpha with an unstable/undocumented result
    // shape; this comparison app cares about LLM generation speed, not STT engine choice.

    // AI Edge RAG SDK — Gecko embedder + vector store for retrieval. Generation still goes
    // through NanoLlmEngine (Gemini Nano/AICore) above, not this SDK's own MediaPipe LLM path.
    implementation("com.google.ai.edge.localagents:localagents-rag:0.1.0")
    implementation("com.google.mediapipe:tasks-genai:0.10.22")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-guava:1.9.0")

    // PDF text extraction for messy/scraped real-world documents (no native code needed —
    // this is a pure-Java port of Apache PDFBox).
    implementation("com.tom-roush:pdfbox-android:2.0.27.0")
}
