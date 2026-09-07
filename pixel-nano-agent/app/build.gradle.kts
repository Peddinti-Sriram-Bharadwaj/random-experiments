plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.google.devtools.ksp")
}

android {
    namespace = "com.example.nanoagent"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.nanoagent"
        minSdk = 31
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

    // Gemini Nano via AICore — same engine as pixel-nano-assistant. Structured Output is what
    // stands in for "function calling" here, since AICore's Prompt API has no native
    // FunctionDeclaration/Tool object (that's exclusive to the MediaPipe/localagents-fc SDK,
    // which only targets self-hosted Gemma via LiteRT, not Nano).
    implementation("com.google.mlkit:genai-prompt:1.0.0-beta4") // beta2 lacks GenerateTypedContentRequest
    ksp("com.google.mlkit:genai-schema-compiler:1.0.0-alpha1")
}
