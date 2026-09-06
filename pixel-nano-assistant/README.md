# Nano RAG

On-device RAG (Retrieval-Augmented Generation) app for Android, built on the Pixel 10 Pro:

- **Retrieval**: Google's [AI Edge RAG SDK](https://developers.google.com/edge/mediapipe/solutions/genai/rag/android) — the `Gecko-110m-en` embedder (Gemma-family, on-device, no cloud calls) + an in-memory vector store.
- **Generation**: Gemini Nano via ML Kit's GenAI Prompt API (AICore) — same engine used in the `pixel-offline-assistant` comparison benchmark, not the RAG SDK's own MediaPipe/Gemma `.task` generation path.
- **Corpus**: bundled as an APK asset (`app/src/main/assets/rag_docs/`), so there's zero external-storage dependency — see "A storage gotcha" below for why that matters.

## Setup

Standard Gradle/Android build, no NDK:

```bash
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

On first launch the app downloads the Gecko embedder model (~443MB) directly from Hugging Face into its own internal storage — keep the app in the foreground/screen on until that finishes (see gotcha below).

## A storage gotcha (and why the model downloads at runtime instead of being adb-pushed)

On this specific device/OS build, files written into the app's **external** files dir via `adb push` or `adb shell mkdir` were *invisible* to the app's own `java.io.File` API at runtime — `exists()`/`listFiles()` silently failed even though the files were genuinely there and POSIX-readable via plain `adb shell`. `run-as <pkg>` confirmed it wasn't a race: the app's own UID got `Permission denied` reading and writing there, on what turned out to be a production **"user" build** (not userdebug/eng), which enforces scoped-storage isolation more strictly than adb tooling assumes.

Fix: skip external storage entirely. The embedder model downloads via plain `HttpURLConnection` straight into **internal** storage (`filesDir`), and the document corpus ships as a bundled APK asset instead of being pushed separately. Both are then guaranteed-accessible without any adb choreography.

## SDK API notes (undocumented at time of writing)

Google's official RAG guide doesn't show the exact `insert`/retrieval method signatures. We extracted them directly from the real `localagents-rag-0.1.0.aar` via `javap -p` on its `classes.jar` (see the parent conversation for the full technique) since the linked GitHub sample repo (`google-ai-edge/ai-edge-apis`) currently 404s:

```java
// com.google.ai.edge.localagents.rag.models.Embedder<T>
ListenableFuture<ImmutableList<Float>> getEmbeddings(EmbeddingRequest<T>)
ListenableFuture<ImmutableList<ImmutableList<Float>>> getBatchEmbeddings(EmbeddingRequest<T>)

// com.google.ai.edge.localagents.rag.memory.VectorStore<T>
void insert(VectorStoreRecord<T>)
List<VectorStoreRecord<T>> getNearestRecords(List<Float> queryEmbedding, int topK, float minScore)

// GeckoEmbeddingModel constructor
GeckoEmbeddingModel(String modelPath, Optional<String> tokenizerPath, boolean useGpu)

// EmbedData.TaskType — asymmetric retrieval embeddings (index docs vs. embed the query differently)
RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, SEMANTIC_SIMILARITY, QUESTION_ANSWERING, ...
```

`DefaultVectorStore<T>` (in-memory, no persistence) and `SqliteVectorStore` (persistent) both implement `VectorStore<T>` with the same two methods above.
