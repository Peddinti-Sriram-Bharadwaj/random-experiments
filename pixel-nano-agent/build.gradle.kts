// KSP is declared via classpath (rather than the plugins{} DSL) per Google's documented
// setup for the Structured Output API's annotation processor — see
// https://developers.google.com/ml-kit/genai/prompt/android/structured-output
buildscript {
    repositories {
        google()
        mavenCentral()
    }
    dependencies {
        classpath("com.google.devtools.ksp:com.google.devtools.ksp.gradle.plugin:2.3.6")
    }
}

plugins {
    id("com.android.application") version "8.7.2" apply false
    id("org.jetbrains.kotlin.android") version "2.2.0" apply false
}
