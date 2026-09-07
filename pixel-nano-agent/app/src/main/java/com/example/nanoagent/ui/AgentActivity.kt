package com.example.nanoagent.ui

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
import com.example.nanoagent.agent.ActionDispatcher
import com.example.nanoagent.agent.NanoAgent
import com.example.nanoagent.databinding.ActivityAgentBinding
import com.google.mlkit.genai.prompt.Generation
import kotlinx.coroutines.launch

class AgentActivity : AppCompatActivity() {

    private lateinit var binding: ActivityAgentBinding
    private lateinit var agent: NanoAgent
    private lateinit var dispatcher: ActionDispatcher

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityAgentBinding.inflate(layoutInflater)
        setContentView(binding.root)

        ViewCompat.setOnApplyWindowInsetsListener(binding.rootLayout) { view, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            view.setPadding(view.paddingLeft, bars.top, view.paddingRight, bars.bottom)
            insets
        }

        agent = NanoAgent(Generation.getClient())
        dispatcher = ActionDispatcher(applicationContext)

        binding.runButton.setOnClickListener {
            val command = binding.commandInput.text.toString()
            if (command.isBlank()) return@setOnClickListener
            runCommand(command)
        }
    }

    private fun runCommand(command: String) {
        binding.runButton.isEnabled = false
        binding.parsedActionText.text = "Interpreting…"
        binding.resultText.text = ""

        lifecycleScope.launch {
            try {
                val action = agent.interpret(command)
                binding.parsedActionText.text = action.toString()

                val result = dispatcher.dispatch(action)
                binding.resultText.text = when (result) {
                    is ActionDispatcher.Result.Handled -> result.message
                    is ActionDispatcher.Result.Failed -> "⚠ ${result.reason}"
                }
            } catch (e: Exception) {
                binding.parsedActionText.text = "(failed to interpret)"
                binding.resultText.text = "⚠ ${e.message}"
            } finally {
                binding.runButton.isEnabled = true
            }
        }
    }
}
