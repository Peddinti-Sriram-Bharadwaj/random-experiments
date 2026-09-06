# Nano Assistant

Same voice-assistant shape as `../pixel-offline-assistant`, but using Google's ML Kit GenAI Prompt API (Gemini Nano via AICore) instead of a hand-rolled llama.cpp/whisper.cpp pipeline. Built specifically to benchmark vendor-optimized on-device inference against the DIY native approach on the same Pixel 10 Pro.

STT uses Android's standard on-device `SpeechRecognizer` rather than ML Kit's (still-alpha) GenAI Speech Recognition API.

## Setup

Standard Gradle/Android build, no NDK or native dependencies:

```bash
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

## Result

Gemini Nano generated a short reply in ~4.1s total vs. llama.cpp/Qwen2.5-1.5B's much slower CPU-bound generation — the gap narrows a lot once llama.cpp's GPU path is fixed (see the sibling project's known issues), and for longer/more complex prompts Nano's latency scales up too (~22s observed for a longer explanation), so the two aren't universally 50x apart in practice.
