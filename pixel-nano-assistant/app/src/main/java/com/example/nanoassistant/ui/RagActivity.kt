package com.example.nanoassistant.ui

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
import com.example.nanoassistant.databinding.ActivityRagBinding
import com.example.nanoassistant.rag.RagPipeline
import com.example.nanoassistant.rag.RagPipelineFactory
import com.example.nanoassistant.rag.ingest.PdfTextExtractor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.net.URL

class RagActivity : AppCompatActivity() {

    private lateinit var binding: ActivityRagBinding
    private lateinit var pipeline: RagPipeline

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
                val (createdPipeline, alreadyIndexed) =
                    RagPipelineFactory.create(this@RagActivity, geckoModel.absolutePath, tokenizer.absolutePath)
                pipeline = createdPipeline

                if (alreadyIndexed) {
                    binding.indexStatus.text = "Loaded persisted index from a previous launch."
                } else {
                    val docNames = assets.list("rag_docs")?.toList().orEmpty()
                    var totalChunks = 0
                    for (name in docNames) {
                        binding.indexStatus.text = "Extracting… $name"
                        val text = withContext(Dispatchers.Default) {
                            if (name.endsWith(".pdf", ignoreCase = true)) {
                                assets.open("rag_docs/$name").use { PdfTextExtractor.extractText(this@RagActivity, it) }
                            } else {
                                assets.open("rag_docs/$name").bufferedReader().use { it.readText() }
                            }
                        }
                        var docChunks = 0
                        pipeline.indexDocument(text, sourceId = name) { indexed ->
                            docChunks = indexed
                            binding.indexStatus.text = "Indexing… $name (${totalChunks + docChunks} chunks so far)"
                        }
                        totalChunks += docChunks
                    }
                    binding.indexStatus.text = "Indexed $totalChunks chunks from ${docNames.size} bundled doc(s)."
                }
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
        binding.rewrittenQueryText.text = ""
        binding.retrievedChunks.text = "Retrieving…"
        binding.refinedContextText.text = "Filtering (System 2 Attention)…"
        binding.answerText.text = "Thinking…"

        lifecycleScope.launch {
            val result = pipeline.ask(query, topK = 3)

            binding.rewrittenQueryText.text = result.rewrittenQuery
            binding.retrievedChunks.text = if (result.rerankedChunks.isEmpty()) {
                "(nothing retrieved)"
            } else {
                result.rerankedChunks.mapIndexed { i, c -> "[$i] (%.2f) %s".format(c.score, c.text) }
                    .joinToString("\n\n")
            }
            binding.refinedContextText.text = result.refinedContext.ifBlank { "(empty)" }
            binding.answerText.text = result.answer
            binding.askButton.isEnabled = true
        }
    }
}
