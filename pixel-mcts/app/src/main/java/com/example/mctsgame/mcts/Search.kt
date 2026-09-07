package com.example.mctsgame.mcts

import com.example.mctsgame.game.Connect4

data class SearchResult(val bestAction: Int, val visitDistribution: Map<Int, Float>)

/**
 * AlphaZero-style MCTS from rootState. Each simulation: SELECT (PUCT) -> EXPAND (one network
 * call, no rollout) -> BACKPROP. Deterministic (argmax over visit counts) since this is for
 * gameplay, not self-play training data generation — no temperature sampling or Dirichlet noise.
 */
fun mctsSearch(
    rootState: Connect4,
    networkFn: (Connect4) -> Pair<FloatArray, Float>,
    numSimulations: Int
): SearchResult {
    val root = Node(state = rootState)
    expand(root, networkFn)

    repeat(numSimulations) {
        val leaf = select(root)
        val value = if (leaf.isTerminal()) leaf.state.reward() else expand(leaf, networkFn)
        backpropagate(leaf, value)
    }

    val totalVisits = root.children.values.sumOf { it.visitCount }
    val pi: Map<Int, Float> = if (totalVisits > 0) {
        root.children.mapValues { (_, child) -> child.visitCount.toFloat() / totalVisits }
    } else {
        root.children.mapValues { (_, child) -> child.priorProb }
    }

    val bestAction = pi.maxBy { it.value }.key
    return SearchResult(bestAction, pi)
}
