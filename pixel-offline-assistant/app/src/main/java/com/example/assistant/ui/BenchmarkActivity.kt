package com.example.assistant.ui

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
import com.example.assistant.databinding.ActivityBenchmarkBinding
import com.example.assistant.engine.LlamaLlmEngine
import com.example.assistant.engine.NanoLlmEngine
import kotlinx.coroutines.launch
import java.io.File

class BenchmarkActivity : AppCompatActivity() {

    private lateinit var binding: ActivityBenchmarkBinding
    private val llama = LlamaLlmEngine()
    private val nano = NanoLlmEngine()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityBenchmarkBinding.inflate(layoutInflater)
        setContentView(binding.root)

        ViewCompat.setOnApplyWindowInsetsListener(binding.rootLayout) { view, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            view.setPadding(view.paddingLeft, bars.top, view.paddingRight, bars.bottom)
            insets
        }

        binding.runButton.isEnabled = false
        binding.llamaResult.text = "Loading models…"

        val modelsDir = File(getExternalFilesDir(null), "models")
        lifecycleScope.launch {
            llama.init(File(modelsDir, "qwen2.5-1.5b-instruct-q4_k_m.gguf").absolutePath)
            nano.init("") // unused for Nano — model lives in AICore
            binding.llamaResult.text = ""
            binding.runButton.isEnabled = true
        }

        binding.runButton.setOnClickListener {
            val prompt = binding.promptInput.text.toString()
            if (prompt.isBlank()) return@setOnClickListener
            runBenchmark(prompt)
        }
    }

    private fun runBenchmark(prompt: String) {
        binding.runButton.isEnabled = false
        binding.llamaResult.text = "Running…"
        binding.nanoResult.text = "Running…"
        binding.llamaTiming.text = ""
        binding.nanoTiming.text = ""

        val chatPrompt = "<|im_start|>user\n$prompt\n<|im_end|>\n<|im_start|>assistant\n"

        var llamaDone = false
        var nanoDone = false
        fun maybeReenableButton() {
            if (llamaDone && nanoDone) binding.runButton.isEnabled = true
        }

        // Each engine updates the UI independently as soon as IT finishes — Nano typically
        // finishes in seconds while llama.cpp is still generating, and the UI should show that.
        lifecycleScope.launch {
            val start = System.currentTimeMillis()
            val sb = StringBuilder()
            llama.generate(chatPrompt).collect { sb.append(it) }
            binding.llamaResult.text = sb.toString()
            binding.llamaTiming.text = "${System.currentTimeMillis() - start} ms"
            llamaDone = true
            maybeReenableButton()
        }
        lifecycleScope.launch {
            val start = System.currentTimeMillis()
            val sb = StringBuilder()
            nano.generate(prompt).collect { sb.append(it) }
            binding.nanoResult.text = sb.toString()
            binding.nanoTiming.text = "${System.currentTimeMillis() - start} ms"
            nanoDone = true
            maybeReenableButton()
        }
    }

    override fun onDestroy() {
        llama.release()
        nano.release()
        super.onDestroy()
    }
}
