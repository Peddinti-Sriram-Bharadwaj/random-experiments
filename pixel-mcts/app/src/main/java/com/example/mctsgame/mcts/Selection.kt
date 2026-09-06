package com.example.mctsgame.mcts

import kotlin.math.max
import kotlin.math.sqrt

private const val C_PUCT = 1.5f // exploration constant; higher = more prior-guided exploration

/** PUCT score. Q is the child's average value negated (parent is the opponent's turn);
 *  U is an exploration bonus weighted by the network's prior probability. */
fun puctScore(parent: Node, child: Node): Float {
    val q = if (child.visitCount > 0) -(child.totalValue / child.visitCount) else 0f
    val u = C_PUCT * child.priorProb * sqrt(max(1, parent.visitCount).toFloat()) / (1 + child.visitCount)
    return q + u
}

/** Walk from 'node' down to a leaf using PUCT scores, stopping at an unexpanded node or a
 *  terminal state. */
fun select(start: Node): Node {
    var node = start
    while (node.isExpanded() && !node.isTerminal()) {
        node = node.children.values.maxBy { puctScore(node, it) }
    }
    return node
}
