package com.example.mctsgame.mcts

/** Walk from 'node' up to the root, updating visitCount and totalValue at every node along
 *  the path. The reward's sign flips at each step, since each parent represents the
 *  opponent's turn relative to the child. */
fun backpropagate(start: Node, initialReward: Float) {
    var node: Node? = start
    var reward = initialReward
    while (node != null) {
        node.visitCount += 1
        node.totalValue += reward
        reward = -reward
        node = node.parent
    }
}
