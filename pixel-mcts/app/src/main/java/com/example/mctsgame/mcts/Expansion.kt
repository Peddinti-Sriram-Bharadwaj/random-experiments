package com.example.mctsgame.mcts

/** Evaluate 'node' with the network, expand all legal children with their prior
 *  probabilities, and return the value estimate for backpropagation. */
fun expand(node: Node, networkFn: (com.example.mctsgame.game.Connect4) -> Pair<FloatArray, Float>): Float {
    val (policyProbs, value) = networkFn(node.state)
    val legal = node.state.legalActions()

    for (action in legal) {
        val nextState = node.state.applyAction(action)
        node.children[action] = Node(
            state = nextState,
            parent = node,
            actionTaken = action,
            priorProb = policyProbs.getOrElse(action) { 0f }
        )
    }
    return value
}
