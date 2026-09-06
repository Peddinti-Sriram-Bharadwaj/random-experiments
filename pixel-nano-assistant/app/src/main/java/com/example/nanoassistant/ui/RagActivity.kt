package com.example.nanoassistant.ui

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
import com.example.nanoassistant.databinding.ActivityRagBinding
import com.example.nanoassistant.rag.RagStore
import com.google.mlkit.genai.prompt.Generation
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.net.URL

class RagActivity : AppCompatActivity() {

    private lateinit var binding: ActivityRagBinding
    private lateinit var ragStore: RagStore
    private val nano = Generation.getClient()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityRagBinding.inflate(layoutInflater)
        setContentView(binding.root)

        ViewCompat.setOnApplyWindowInsetsListener(binding.rootLayout) { view, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            view.setPadding(view.paddingLeft, bars.top, view.paddingRight, bars.bottom)
            insets
        }

        binding.askButton.isEnabled = false
        binding.indexStatus.text = "Preparing embedder…"

        // Internal storage only — this device's scoped-storage enforcement blocks the app's
        // own java.io.File reads of its external files dir when the files were written by adb/
        // shell, even though POSIX permissions look fine. Internal filesDir has no such issue.
        val modelsDir = File(filesDir, "models").apply { mkdirs() }

        lifecycleScope.launch {
            try {
                val geckoModel = downloadIfNeeded(
                    modelsDir, "Gecko_256_f32.tflite",
                    "https://huggingface.co/litert-community/Gecko-110m-en/resolve/main/Gecko_256_f32.tflite"
                )
                val tokenizer = downloadIfNeeded(
                    modelsDir, "sentencepiece.model",
                    "https://huggingface.co/litert-community/Gecko-110m-en/resolve/main/sentencepiece.model"
                )

                binding.indexStatus.text = "Loading embedder…"
                ragStore = RagStore(geckoModel.absolutePath, tokenizer.absolutePath)

                val docNames = assets.list("rag_docs")?.toList().orEmpty()
                var chunkCount = 0
                for (name in docNames) {
                    val text = assets.open("rag_docs/$name").bufferedReader().use { it.readText() }
                    val chunks = ragStore.chunk(text)
                    for (chunk in chunks) {
                        ragStore.indexChunk(chunk)
                        chunkCount++
                        binding.indexStatus.text = "Indexing… $name ($chunkCount chunks so far)"
                    }
                }
                binding.indexStatus.text = "Indexed $chunkCount chunks from ${docNames.size} bundled doc(s)."
                binding.askButton.isEnabled = true
            } catch (e: Exception) {
                binding.indexStatus.text = "Setup failed: ${e.message} — tap RAG button again to retry."
            }
        }

        binding.askButton.setOnClickListener {
            val query = binding.queryInput.text.toString()
            if (query.isBlank()) return@setOnClickListener
            askQuestion(query)
        }
    }

    private suspend fun downloadIfNeeded(dir: File, filename: String, url: String): File =
        withContext(Dispatchers.IO) {
            val dest = File(dir, filename)
            if (dest.exists() && dest.length() > 0) return@withContext dest

            val tmp = File(dir, "$filename.part")
            var lastError: Exception? = null
            repeat(5) { attempt ->
                try {
                    withContext(Dispatchers.Main) {
                        binding.indexStatus.text = "Downloading $filename… (attempt ${attempt + 1}/5) " +
                            "— keep the app in the foreground until this finishes"
                    }
                    val connection = URL(url).openConnection().apply {
                        connectTimeout = 15_000
                        readTimeout = 15_000 // fail fast on a stalled socket (e.g. Doze) instead of hanging
                    }
                    connection.getInputStream().use { input ->
                        tmp.outputStream().use { output -> input.copyTo(output) }
                    }
                    tmp.renameTo(dest)
                    return@withContext dest
                } catch (e: Exception) {
                    lastError = e
                }
            }
            throw lastError!!
        }

    private fun askQuestion(query: String) {
        binding.askButton.isEnabled = false
        binding.retrievedChunks.text = "Retrieving…"
        binding.answerText.text = "Thinking…"

        lifecycleScope.launch {
            val chunks = ragStore.retrieve(query, topK = 3)
            binding.retrievedChunks.text = if (chunks.isEmpty()) {
                "(nothing retrieved)"
            } else {
                chunks.mapIndexed { i, c -> "[$i] $c" }.joinToString("\n\n")
            }

            val context = chunks.joinToString("\n\n")
            val prompt = """
                Context:
                $context

                Question: $query
                Answer using only the context above. If the context doesn't contain the answer, say so.
            """.trimIndent()

            val response = nano.generateContent(prompt)
            binding.answerText.text = response.candidates.firstOrNull()?.text.orEmpty()
            binding.askButton.isEnabled = true
        }
    }
}
