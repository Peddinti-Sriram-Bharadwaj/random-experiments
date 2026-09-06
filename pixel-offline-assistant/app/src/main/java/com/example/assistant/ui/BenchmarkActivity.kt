package com.example.assistant.ui

import android.content.Context
import android.os.BatteryManager
import android.os.Bundle
import android.os.Process
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

    private fun batteryManager() = getSystemService(Context.BATTERY_SERVICE) as BatteryManager

    private fun runBenchmark(prompt: String) {
        binding.runButton.isEnabled = false
        binding.llamaResult.text = "Running…"
        binding.nanoResult.text = "Running…"
        binding.llamaTiming.text = ""
        binding.nanoTiming.text = ""
        binding.tableLlamaWall.text = "—"
        binding.tableLlamaCpu.text = "—"
        binding.tableLlamaCores.text = "—"
        binding.tableNanoWall.text = "—"

        val bm = batteryManager()
        val pctBefore = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
        val chargeBefore = bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)
        binding.batteryPctBefore.text = "$pctBefore%"
        binding.chargeCounterBefore.text = "${chargeBefore / 1000} mAh"
        binding.batteryPctAfter.text = "—"
        binding.chargeCounterAfter.text = "—"
        binding.batteryNote.text = if (bm.isCharging()) {
            "Device is charging — charge-counter deltas reflect charging input minus consumption, not pure draw. Unplug for a clean reading."
        } else ""

        val chatPrompt = "<|im_start|>user\n$prompt\n<|im_end|>\n<|im_start|>assistant\n"

        var llamaDone = false
        var nanoDone = false
        fun maybeFinish() {
            if (llamaDone && nanoDone) {
                binding.runButton.isEnabled = true
                binding.batteryPctAfter.text = "${bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)}%"
                binding.chargeCounterAfter.text =
                    "${bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER) / 1000} mAh"
            }
        }

        // Each engine updates the UI independently as soon as IT finishes — Nano typically
        // finishes in seconds while llama.cpp is still generating, and the UI should show that.
        lifecycleScope.launch {
            val wallStart = System.currentTimeMillis()
            val cpuStart = Process.getElapsedCpuTime()
            val sb = StringBuilder()
            llama.generate(chatPrompt).collect { sb.append(it) }
            val wallMs = System.currentTimeMillis() - wallStart
            val cpuMs = Process.getElapsedCpuTime() - cpuStart

            binding.llamaResult.text = sb.toString()
            binding.llamaTiming.text = "$wallMs ms"
            binding.tableLlamaWall.text = "$wallMs ms"
            binding.tableLlamaCpu.text = "$cpuMs ms"
            binding.tableLlamaCores.text = "%.2f".format(cpuMs.toDouble() / wallMs)
            llamaDone = true
            maybeFinish()
        }
        lifecycleScope.launch {
            val start = System.currentTimeMillis()
            val sb = StringBuilder()
            nano.generate(prompt).collect { sb.append(it) }
            val wallMs = System.currentTimeMillis() - start
            binding.nanoResult.text = sb.toString()
            binding.nanoTiming.text = "$wallMs ms"
            binding.tableNanoWall.text = "$wallMs ms"
            nanoDone = true
            maybeFinish()
        }
    }

    override fun onDestroy() {
        llama.release()
        nano.release()
        super.onDestroy()
    }
}
