# Offline Assistant

Fully on-device voice assistant for Android (built/tested on a Pixel 10 Pro): mic → whisper.cpp (STT) → llama.cpp/Qwen2.5-1.5B-Instruct (LLM) → Android TTS. No network calls in the critical path.

## Setup

This repo excludes the vendored native dependencies and model weights (too large for git). Before building:

```bash
# 1. Fetch the native engines at the commits this was built against
git clone https://github.com/ggml-org/llama.cpp.git app/src/main/cpp/llama.cpp
git clone https://github.com/ggml-org/whisper.cpp.git app/src/main/cpp/whisper.cpp

# 2. Download model weights into ./models/
mkdir -p models
curl -L -o models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
curl -L -o models/ggml-base.en.bin \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
curl -L -o models/ggml-tiny.en.bin \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin"

# 3. Push models onto a connected device (app-private external storage)
adb push models/*.bin models/*.gguf \
  /sdcard/Android/data/com.example.assistant/files/models/
```

Requires the Vulkan-Hpp/SPIRV-Headers Homebrew packages on macOS to build the (currently CPU-forced) GPU backend — see `app/build.gradle.kts` CMake args.

## Known issues

- GPU (Vulkan) path for whisper/llama.cpp produces garbage output on this device's GPU — Tensor G5's Imagination PowerVR DXT-48-1536 (not Mali; Tensor G5 dropped Mali/Immortalis after Tensor G1-G4) — when quantized-model cooperative-matrix extensions are enabled. Worked around by disabling `GGML_VK_DISABLE_COOPMAT`/`COOPMAT2`/`INTEGER_DOT_PRODUCT` at runtime (see `llama_jni.cpp`), which fixed llama.cpp but whisper.cpp is still forced to CPU. See also [ggml-org/llama.cpp#28214](https://github.com/ggml-org/llama.cpp/issues/28214) — a related but distinct Imagination PowerVR (Tensor G6, different sub-series) Vulkan crash, suggesting a broader pattern with this GPU vendor's driver on ggml-vulkan.
- CPU-only inference is slow (~5-6s/token for the LLM in the worst case observed).

See `pixel-nano-assistant/` for a comparison app using Gemini Nano/AICore instead — dramatically faster on the same hardware since it uses a dedicated accelerator rather than raw CPU/GPU compute.
