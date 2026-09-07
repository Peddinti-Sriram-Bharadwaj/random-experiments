package com.example.mctsgame.game

/**
 * Connect-4 game state, ported 1:1 from the Python reference implementation
 * (games/connect4.py) in the neural_sor MCTS project. Immutable: apply()
 * returns a new state rather than mutating this one.
 *
 * Board is a flat list of 42 cells (6 rows x 7 cols), indices 0-41:
 *   0  1  2  3  4  5  6   (top row, row 0)
 *   ...
 *   35 36 37 38 39 40 41  (bottom row, row 5)
 * Each cell is 'X', 'O', or null (empty).
 */
class Connect4 private constructor(
    val board: Array<Char?>,
    val currentPlayer: Char
) {
    companion object {
        const val ROWS = 6
        const val COLS = 7
        const val NUM_ACTIONS = COLS

        private val WIN_LINES: List<IntArray> = buildList {
            // Horizontal
            for (r in 0 until ROWS) for (c in 0..COLS - 4) {
                val idx = r * COLS + c
                add(intArrayOf(idx, idx + 1, idx + 2, idx + 3))
            }
            // Vertical
            for (r in 0..ROWS - 4) for (c in 0 until COLS) {
                val idx = r * COLS + c
                add(intArrayOf(idx, idx + COLS, idx + 2 * COLS, idx + 3 * COLS))
            }
            // Diagonal /
            for (r in 3 until ROWS) for (c in 0..COLS - 4) {
                val idx = r * COLS + c
                add(intArrayOf(idx, idx - COLS + 1, idx - 2 * COLS + 2, idx - 3 * COLS + 3))
            }
            // Diagonal \
            for (r in 0..ROWS - 4) for (c in 0..COLS - 4) {
                val idx = r * COLS + c
                add(intArrayOf(idx, idx + COLS + 1, idx + 2 * COLS + 2, idx + 3 * COLS + 3))
            }
        }

        fun initial(): Connect4 = Connect4(arrayOfNulls(ROWS * COLS), 'X')
    }

    fun legalActions(): List<Int> = (0 until COLS).filter { board[it] == null }

    fun applyAction(action: Int): Connect4 {
        require(board[action] == null) { "Column $action is full" }
        val newBoard = board.copyOf()
        for (r in ROWS - 1 downTo 0) {
            val idx = r * COLS + action
            if (newBoard[idx] == null) {
                newBoard[idx] = currentPlayer
                break
            }
        }
        val nextPlayer = if (currentPlayer == 'X') 'O' else 'X'
        return Connect4(newBoard, nextPlayer)
    }

    fun winner(): Char? {
        for (line in WIN_LINES) {
            val (a, b, c, d) = line
            val v = board[a]
            if (v != null && v == board[b] && v == board[c] && v == board[d]) return v
        }
        return null
    }

    fun isTerminal(): Boolean = winner() != null || board.all { it != null }

    /** Reward from currentPlayer's perspective. The winner (if any) is always
     *  the player who just moved, i.e. never currentPlayer. */
    fun reward(): Float = if (winner() != null) -1f else 0f

    /** Two binary planes from the current player's perspective, flattened:
     *  plane 0 = current player's pieces, plane 1 = opponent's pieces. */
    fun toArray(): FloatArray {
        val opponent = if (currentPlayer == 'X') 'O' else 'X'
        val out = FloatArray(2 * ROWS * COLS)
        for (i in board.indices) {
            if (board[i] == currentPlayer) out[i] = 1f
            if (board[i] == opponent) out[ROWS * COLS + i] = 1f
        }
        return out
    }
}

private operator fun IntArray.component1() = this[0]
private operator fun IntArray.component2() = this[1]
private operator fun IntArray.component3() = this[2]
private operator fun IntArray.component4() = this[3]
