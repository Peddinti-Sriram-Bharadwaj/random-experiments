package com.example.assistant.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import com.example.assistant.AssistantState
import com.example.assistant.AssistantViewModel
import com.example.assistant.databinding.ActivityMainBinding
import com.example.assistant.engine.AudioCapture
import com.example.assistant.engine.LlamaLlmEngine
import com.example.assistant.engine.WhisperSttEngine
import kotlinx.coroutines.launch
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var viewModel: AssistantViewModel
    private lateinit var modelsDir: File
    private val audioCapture = AudioCapture()
    private var isRecording = false

    private val requestMic = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (!granted) binding.stateLabel.text = "Mic permission denied"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // targetSdk 35 draws edge-to-edge by default, so the root layout starts under the
        // status bar / camera cutout unless we apply those insets as padding ourselves.
        ViewCompat.setOnApplyWindowInsetsListener(binding.rootLayout) { view, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            view.setPadding(view.paddingLeft, bars.top, view.paddingRight, bars.bottom)
            insets
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            requestMic.launch(Manifest.permission.RECORD_AUDIO)
        }

        modelsDir = File(getExternalFilesDir(null), "models")
        viewModel = ViewModelProvider(
            this,
            AssistantViewModelFactory(
                sttEngine = WhisperSttEngine(),
                llmEngine = LlamaLlmEngine(),
                speak = { text -> TtsHelper.speak(this, text) },
                sttModelPath = File(modelsDir, "ggml-base.en.bin").absolutePath,
                llmModelPath = File(modelsDir, "qwen2.5-1.5b-instruct-q4_k_m.gguf").absolutePath
            )
        )[AssistantViewModel::class.java]

        binding.whisperButton.setOnClickListener {
            viewModel.switchSttEngine(
                "Whisper tiny.en",
                WhisperSttEngine(),
                File(modelsDir, "ggml-tiny.en.bin").absolutePath
            )
        }
        binding.parakeetButton.setOnClickListener {
            viewModel.switchSttEngine(
                "Whisper base.en",
                WhisperSttEngine(),
                File(modelsDir, "ggml-base.en.bin").absolutePath
            )
        }

        binding.micButton.setOnClickListener {
            if (!isRecording) {
                isRecording = true
                viewModel.onMicTapped()
                lifecycleScope.launch {
                    val pcm = audioCapture.recordUntil { !isRecording }
                    // Button stays disabled (see render()) until the ViewModel reaches Idle again —
                    // prevents a second whisper_full/llama call overlapping the first on the same
                    // native context, which is not thread-safe and previously crashed with SIGSEGV.
                    viewModel.onSpeechCaptured(pcm)
                }
            } else {
                isRecording = false
            }
        }

        binding.benchmarkButton.setOnClickListener {
            startActivity(Intent(this, BenchmarkActivity::class.java))
        }

        lifecycleScope.launch {
            viewModel.state.collect { state -> render(state) }
        }
        lifecycleScope.launch {
            viewModel.sttEngineName.collect { name -> binding.sttEngineLabel.text = "STT: $name" }
        }
    }

    private fun render(state: AssistantState) {
        binding.stateLabel.text = state::class.simpleName
        when (state) {
            is AssistantState.Thinking -> binding.transcriptView.text = state.partialResponse
            is AssistantState.Speaking -> binding.transcriptView.text = state.fullResponse
            is AssistantState.Error -> binding.transcriptView.text = state.message
            is AssistantState.SwitchingModel -> binding.transcriptView.text = "Loading ${state.name}…"
            is AssistantState.Listening -> binding.transcriptView.text = ""
            else -> Unit
        }

        val idle = state is AssistantState.Idle
        binding.micButton.isEnabled = idle || state is AssistantState.Listening
        binding.micButton.text = when {
            state is AssistantState.Listening -> "Tap to stop"
            idle -> "Tap to talk"
            else -> "Working…"
        }
        binding.whisperButton.isEnabled = idle
        binding.parakeetButton.isEnabled = idle
        if (state is AssistantState.Idle || state is AssistantState.Error) {
            isRecording = false
        }
    }
}
