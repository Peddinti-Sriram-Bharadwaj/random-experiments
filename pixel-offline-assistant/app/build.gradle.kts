plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.assistant"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.assistant"
        minSdk = 33          // Pixel 10 Pro ships well above this; keeps NDK/AudioRecord APIs simple
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"

        ndk {
            // Tensor G5 is arm64 only for our purposes; skip x86 to cut build time
            abiFilters += "arm64-v8a"
        }

        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
                cppFlags += "-I/opt/homebrew/opt/vulkan-headers/include"
                cppFlags += "-I/opt/homebrew/opt/spirv-headers/include"
                arguments += listOf(
                    "-DLLAMA_BUILD_EXAMPLES=OFF",
                    "-DLLAMA_BUILD_TESTS=OFF",
                    "-DLLAMA_BUILD_SERVER=OFF",
                    "-DGGML_LLAMAFILE=OFF",
                    "-DGGML_VULKAN=ON",
                    "-DVulkan_GLSLC_EXECUTABLE=/opt/homebrew/bin/glslc",
                    "-DCMAKE_PREFIX_PATH=/opt/homebrew",
                    "-DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH",
                    "-DSPIRV-Headers_DIR=/opt/homebrew/share/cmake/SPIRV-Headers",
                    "-DSPIRV-Tools_DIR=/opt/homebrew/lib/cmake/SPIRV-Tools"
                )
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
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

    ndkVersion = "29.0.14206865" // already installed locally; r27 LTS would need Android Studio's SDK Manager first
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.8.7")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("com.google.android.material:material:1.12.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
    implementation("com.google.mlkit:genai-prompt:1.0.0-beta2")
}
