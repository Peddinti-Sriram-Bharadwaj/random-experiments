package com.example.mctsgame.ui

import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.widget.Button
import android.widget.LinearLayout
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.mctsgame.databinding.ActivityMainBinding
import com.example.mctsgame.game.Connect4
import com.example.mctsgame.mcts.mctsSearch
import com.example.mctsgame.net.AlphaZeroNet
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private const val NUM_SIMULATIONS = 200
private const val HUMAN = 'X'
private const val AI = 'O'

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var net: AlphaZeroNet
    private lateinit var cells: Array<Button>
    private var state = Connect4.initial()
    private var busy = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        net = AlphaZeroNet(this)
        binding.boardArea.viewTreeObserver.addOnGlobalLayoutListener(object : android.view.ViewTreeObserver.OnGlobalLayoutListener {
            override fun onGlobalLayout() {
                binding.boardArea.viewTreeObserver.removeOnGlobalLayoutListener(this)
                val availableWidth = binding.boardArea.width
                val availableHeight = binding.boardArea.height
                val cellSize = minOf(availableWidth / Connect4.COLS, availableHeight / Connect4.ROWS)
                binding.boardFrame.layoutParams = android.widget.FrameLayout.LayoutParams(
                    cellSize * Connect4.COLS, cellSize * Connect4.ROWS
                )
                buildBoard()
                render()
            }
        })

        binding.newGameButton.setOnClickListener {
            state = Connect4.initial()
            busy = false
            render()
        }
    }

    private fun buildBoard() {
        binding.boardGrid.removeAllViews()
        binding.boardGrid.orientation = LinearLayout.VERTICAL
        cells = Array(Connect4.ROWS * Connect4.COLS) { i -> Button(this) }
        val margin = 6

        for (row in 0 until Connect4.ROWS) {
            val rowLayout = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f
                )
            }
            for (col in 0 until Connect4.COLS) {
                val i = row * Connect4.COLS + col
                val cell = cells[i]
                cell.layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, 1f).apply {
                    setMargins(margin, margin, margin, margin)
                }
                cell.text = ""
                cell.background = discDrawable(EMPTY_COLOR)
                cell.setOnClickListener { onColumnTapped(col) }
                rowLayout.addView(cell)
            }
            binding.boardGrid.addView(rowLayout)
        }
    }

    private fun onColumnTapped(column: Int) {
        if (busy || state.isTerminal() || state.currentPlayer != HUMAN) return
        if (column !in state.legalActions()) return

        state = state.applyAction(column)
        render()
        maybeTriggerAiMove()
    }

    private fun maybeTriggerAiMove() {
        if (state.isTerminal() || state.currentPlayer != AI) return
        busy = true
        render()

        lifecycleScope.launch {
            val action = withContext(Dispatchers.Default) {
                mctsSearch(state, net::forward.toNetworkFn(net), NUM_SIMULATIONS).bestAction
            }
            state = state.applyAction(action)
            busy = false
            render()
        }
    }

    private fun discDrawable(color: Int): GradientDrawable =
        GradientDrawable().apply {
            shape = GradientDrawable.OVAL
            setColor(color)
        }

    private fun render() {
        for (i in cells.indices) {
            val color = when (state.board[i]) {
                'X' -> HUMAN_COLOR
                'O' -> AI_COLOR
                else -> EMPTY_COLOR
            }
            cells[i].background = discDrawable(color)
            cells[i].isEnabled = !busy
        }
        binding.boardGrid.alpha = if (busy) 0.6f else 1f

        binding.statusText.text = when {
            state.winner() == HUMAN -> "You win!"
            state.winner() == AI -> "AI wins."
            state.isTerminal() -> "Draw."
            busy -> "AI is thinking…"
            state.currentPlayer == HUMAN -> "Your turn (X)."
            else -> "AI is thinking…"
        }
    }

    private companion object {
        val EMPTY_COLOR = Color.rgb(10, 18, 40)
        val HUMAN_COLOR = Color.rgb(230, 57, 70)
        val AI_COLOR = Color.rgb(255, 190, 40)
    }
}

/** Adapts AlphaZeroNet's raw forward pass into the (state) -> (policyProbs, value) shape
 *  the MCTS expansion step expects, masking illegal actions the same way the Python
 *  network/inference.py wrapper does. */
private fun ((FloatArray) -> Pair<FloatArray, Float>).toNetworkFn(net: AlphaZeroNet):
        (Connect4) -> Pair<FloatArray, Float> = { connect4State ->
    val (logits, value) = this(connect4State.toArray())
    val probs = net.policyProbs(logits, connect4State.legalActions())
    probs to value
}
