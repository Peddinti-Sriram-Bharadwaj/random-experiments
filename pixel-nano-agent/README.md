# Nano Agent

Offline agentic device actions on Android via Gemini Nano/AICore. A natural-language command
("turn on the flashlight", "set an alarm for 7am", "open the camera app") is turned into a real
Android side effect, entirely on-device — no cloud, no MediaPipe/LiteRT custom weights, just the
ML Kit GenAI Prompt API's Structured Output feature.

## Why this isn't "real" function calling — and what it is instead

AICore's Prompt API (`com.google.mlkit:genai-prompt`) has **no native `FunctionDeclaration`/`Tool`
object**. That API exists — `com.google.ai.edge.localagents:localagents-fc` — but it's paired
with the MediaPipe LLM Inference API (`com.google.mediapipe:tasks-genai`) for **self-hosted Gemma
weights via LiteRT**, not for Gemini Nano/AICore. Confirmed by decompiling both `.aar`s directly
(`javap -p` on `classes.jar`, same technique used for `pixel-nano-assistant`'s SDK notes):
`GenerativeModel` in `genai-prompt` exposes `generateContent`/`generateContentStream` and, as of
`1.0.0-beta4`, `generateContent(GenerateTypedContentRequest<T>)` — no `FunctionDeclaration`
anywhere in the jar.

So "function calling" here is built the same way agent frameworks did it before native tool-calling
existed: the model is constrained to emit **one instance of a fixed schema**
(`AgentAction` — a discriminated union over the actions this app supports), and
[`ActionDispatcher`](app/src/main/java/com/example/nanoagent/agent/ActionDispatcher.kt) is the
*only* code that turns that object into a real side effect. The model never names a function or a
code path directly — it can only ever pick one of the six `action` enum values `AgentAction`
declares, and every field it can influence is explicitly typed. That's a meaningfully smaller
attack surface than true tool-calling would be: there's no way for a hallucinated or manipulated
response to reach any Android API this schema doesn't expose a field for.

## SDK version note

`genai-prompt:1.0.0-beta2` (what `pixel-nano-assistant` pins) has **no** Structured Output API at
all — no `GenerateTypedContentRequest`, no typed candidate. That only appears starting in
`1.0.0-beta4` (`GenerateTypedContentRequest`/`GenerateTypedContentResponse`/`TypedCandidate`),
confirmed the same way, by decompiling both versions' `.aar`s side by side. This app pins beta4.

## A real, verified device limitation

On this Pixel 10 Pro, calling the real `generateTypedContentRequest` path throws immediately with
**"Structured output feature is not available"** — and produces **zero logcat output**, meaning
the failure is a client-side capability check that never even reaches AICore. The SDK classes are
correct (compiles and links fine against beta4), but the *on-device* AICore/Nano install hasn't
rolled the Structured Output feature out to this device yet, independent of which SDK version the
app links. This is presumably a staged server-side/AICore-module rollout, not something fixable
from application code.

**Fix: fail open, like every Nano-backed stage in the sibling projects.** `NanoAgent.interpret()`
tries the real typed request first (strictly better when available — schema-enforced at the
AICore layer, not just prompted for), and on any failure falls back to prompting Nano for raw
JSON matching the same shape, parsed by hand with `org.json`. Verified end-to-end on-device: "turn
on the flashlight" → the fallback path parses `{"action": "TOGGLE_FLASHLIGHT", "flashlightOn":
true, ...}` and `ActionDispatcher` calls `CameraManager.setTorchMode` for real.

## Supported actions

| Action | Android mechanism | Confirmation UI? |
|---|---|---|
| `SET_ALARM` | `Intent(AlarmClock.ACTION_SET_ALARM)` | Yes — `EXTRA_SKIP_UI=false` always |
| `SET_TIMER` | `Intent(AlarmClock.ACTION_SET_TIMER)` | Yes — same |
| `CREATE_CALENDAR_EVENT` | `Intent(ACTION_INSERT, CalendarContract.Events.CONTENT_URI)` | Yes — lands in the calendar app's own editor |
| `TOGGLE_FLASHLIGHT` | `CameraManager.setTorchMode` | No — fully reversible, low-stakes |
| `OPEN_APP` | `PackageManager.getLaunchIntentForPackage` (fuzzy label match) | N/A — just launches the app |
| `NONE` | no-op | — |

**Deliberate safety default:** every action that creates or modifies something (alarm, timer,
calendar event) uses `EXTRA_SKIP_UI = false`, so it always lands the user in the target app's own
confirmation screen rather than silently completing. A misinterpreted command costs an extra tap,
never an unwanted alarm or calendar entry. `TOGGLE_FLASHLIGHT` is the only action that completes
without a confirmation screen — that's fine because it's trivially reversible and has no data
consequence.

## Setup

```bash
./gradlew :app:installDebug
```

No model downloads — AICore manages Gemini Nano's model delivery itself, same as
`pixel-nano-assistant`. Requires a device with AICore/Gemini Nano support (Pixel 8+, or other
AICore-supported flagship).

## Relationship to the other two on-device AI projects in this repo

- `pixel-nano-assistant` — RAG over documents, same AICore/Nano engine, earlier `genai-prompt`
  beta (no structured output).
- `pixel-offline-assistant` — self-hosted llama.cpp/whisper.cpp via JNI, the actual alternative
  to AICore when you need model/backend control AICore doesn't expose (see its README's GPU
  workaround notes).
- `pixel-nano-agent` (this project) — same AICore engine as the RAG app, but agentic: the model
  chooses actions, not just text.
