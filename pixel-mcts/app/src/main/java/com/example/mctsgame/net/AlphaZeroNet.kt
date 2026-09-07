package com.example.mctsgame.net

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.exp
import kotlin.math.tanh

/**
 * Reimplementation of the trained AlphaZeroNet forward pass (network/model.py) using plain
 * float arrays instead of JAX/Flax — the model is tiny (~15K params for Connect-4 at
 * hidden_size=128), so hand-written matmuls are simpler and more robust than bundling a full
 * ML runtime (TFLite/ONNX) for two Linear+ReLU layers and two output heads.
 *
 * Architecture: fc1 -> relu -> fc2 -> relu -> {policy_head logits, value_head -> tanh}
 */
class AlphaZeroNet(context: Context, assetName: String = "connect4_weights.json") {

    val inputSize: Int
    val numActions: Int
    val hiddenSize: Int

    private val fc1Kernel: Array<FloatArray> // [inputSize][hiddenSize]
    private val fc1Bias: FloatArray
    private val fc2Kernel: Array<FloatArray> // [hiddenSize][hiddenSize]
    private val fc2Bias: FloatArray
    private val policyKernel: Array<FloatArray> // [hiddenSize][numActions]
    private val policyBias: FloatArray
    private val valueKernel: Array<FloatArray> // [hiddenSize][1]
    private val valueBias: FloatArray

    init {
        val json = context.assets.open(assetName).bufferedReader().use { it.readText() }
        val obj = JSONObject(json)
        inputSize = obj.getInt("input_size")
        numActions = obj.getInt("num_actions")
        hiddenSize = obj.getInt("hidden_size")

        fc1Kernel = obj.getJSONArray("fc1_kernel").to2DFloatArray()
        fc1Bias = obj.getJSONArray("fc1_bias").to1DFloatArray()
        fc2Kernel = obj.getJSONArray("fc2_kernel").to2DFloatArray()
        fc2Bias = obj.getJSONArray("fc2_bias").to1DFloatArray()
        policyKernel = obj.getJSONArray("policy_head_kernel").to2DFloatArray()
        policyBias = obj.getJSONArray("policy_head_bias").to1DFloatArray()
        valueKernel = obj.getJSONArray("value_head_kernel").to2DFloatArray()
        valueBias = obj.getJSONArray("value_head_bias").to1DFloatArray()
    }

    /** Returns (policyLogits, value) for a flattened board encoding of size [inputSize]. */
    fun forward(x: FloatArray): Pair<FloatArray, Float> {
        val h1 = relu(linear(x, fc1Kernel, fc1Bias))
        val h2 = relu(linear(h1, fc2Kernel, fc2Bias))
        val policyLogits = linear(h2, policyKernel, policyBias)
        val value = tanh(linear(h2, valueKernel, valueBias)[0].toDouble()).toFloat()
        return policyLogits to value
    }

    /** Softmax over policyLogits, masking illegal actions to zero probability. */
    fun policyProbs(policyLogits: FloatArray, legalActions: List<Int>): FloatArray {
        val legalSet = legalActions.toHashSet()
        val masked = FloatArray(policyLogits.size) { i ->
            if (i in legalSet) policyLogits[i] else Float.NEGATIVE_INFINITY
        }
        val max = masked.filter { it.isFinite() }.maxOrNull() ?: 0f
        val exps = masked.map { if (it.isFinite()) exp((it - max).toDouble()).toFloat() else 0f }
        val sum = exps.sum()
        return if (sum > 0f) FloatArray(exps.size) { exps[it] / sum } else masked.map { 0f }.toFloatArray()
    }

    private fun linear(x: FloatArray, kernel: Array<FloatArray>, bias: FloatArray): FloatArray {
        val out = bias.copyOf()
        for (i in x.indices) {
            val xi = x[i]
            if (xi == 0f) continue
            val row = kernel[i]
            for (j in row.indices) out[j] += xi * row[j]
        }
        return out
    }

    private fun relu(x: FloatArray): FloatArray = FloatArray(x.size) { if (x[it] > 0f) x[it] else 0f }
}

private fun JSONArray.to1DFloatArray(): FloatArray = FloatArray(length()) { getDouble(it).toFloat() }

private fun JSONArray.to2DFloatArray(): Array<FloatArray> =
    Array(length()) { i -> getJSONArray(i).to1DFloatArray() }
