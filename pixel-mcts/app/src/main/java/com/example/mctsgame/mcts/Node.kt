package com.example.mctsgame.mcts

import com.example.mctsgame.game.Connect4

class Node(
    val state: Connect4,
    val parent: Node? = null,
    val actionTaken: Int? = null,
    var priorProb: Float = 0f
) {
    val children: MutableMap<Int, Node> = mutableMapOf()
    var visitCount: Int = 0
    var totalValue: Float = 0f

    fun isExpanded(): Boolean = children.isNotEmpty()
    fun isTerminal(): Boolean = state.isTerminal()
}
