package com.example.nanoassistant.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import com.example.nanoassistant.AssistantState
import com.example.nanoassistant.AssistantViewModel
import com.example.nanoassistant.databinding.ActivityMainBinding
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var viewModel: AssistantViewModel
    private lateinit var speechRecognizer: SpeechRecognizer

    private val requestMic = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (!granted) binding.stateLabel.text = "Mic permission denied"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

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

        viewModel = ViewModelProvider(
            this,
            AssistantViewModelFactory(speak = { text -> TtsHelper.speak(this, text) })
        )[AssistantViewModel::class.java]

        // Prefer the on-device recognizer where available — genuinely offline on Pixel hardware,
        // same spirit as the llama.cpp/whisper.cpp app, just via Google's own stack.
        val onDeviceAvailable = SpeechRecognizer.isOnDeviceRecognitionAvailable(this)
        speechRecognizer = if (onDeviceAvailable) {
            SpeechRecognizer.createOnDeviceSpeechRecognizer(this)
        } else {
            SpeechRecognizer.createSpeechRecognizer(this)
        }
        speechRecognizer.setRecognitionListener(object : RecognitionListener {
            override fun onResults(results: Bundle) {
                val text = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    ?.firstOrNull().orEmpty()
                viewModel.onTranscript(text)
            }
            override fun onError(error: Int) {
                viewModel.onTranscriptError("Speech recognition error code $error")
            }
            override fun onReadyForSpeech(params: Bundle?) = Unit
            override fun onBeginningOfSpeech() = Unit
            override fun onRmsChanged(rmsdB: Float) = Unit
            override fun onBufferReceived(buffer: ByteArray?) = Unit
            override fun onEndOfSpeech() = Unit
            override fun onPartialResults(partialResults: Bundle?) = Unit
            override fun onEvent(eventType: Int, params: Bundle?) = Unit
        })

        binding.micButton.setOnClickListener {
            viewModel.onMicTapped()
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
            }
            speechRecognizer.startListening(intent)
        }

        lifecycleScope.launch {
            viewModel.state.collect { state -> render(state) }
        }
    }

    private fun render(state: AssistantState) {
        binding.stateLabel.text = state::class.simpleName
        when (state) {
            is AssistantState.Thinking -> binding.transcriptView.text = state.partialResponse
            is AssistantState.Speaking -> {
                binding.transcriptView.text = state.fullResponse
                binding.timingLabel.text = "Generation took ${state.timingMs} ms"
            }
            is AssistantState.Error -> binding.transcriptView.text = state.message
            is AssistantState.Listening -> binding.transcriptView.text = ""
            else -> Unit
        }
        val idle = state is AssistantState.Idle
        binding.micButton.isEnabled = idle
        binding.micButton.text = if (idle) "Tap to talk" else "Working…"
    }

    override fun onDestroy() {
        speechRecognizer.destroy()
        super.onDestroy()
    }
}
