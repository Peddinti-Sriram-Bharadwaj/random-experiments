package com.example.nanoagent.agent

import com.google.mlkit.genai.prompt.GenerateContentRequest
import com.google.mlkit.genai.prompt.GenerativeModel
import com.google.mlkit.genai.prompt.TextPart
import com.google.mlkit.genai.prompt.generateTypedContentRequest
import org.json.JSONObject
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

/**
 * Translates a natural-language command into an [AgentAction]. This is the closest AICore gets
 * to "function calling": there's no native FunctionDeclaration/Tool object on this SDK (that
 * only exists in the separate MediaPipe/localagents-fc stack for self-hosted LiteRT models) —
 * instead, the model is constrained to emit one instance of a fixed schema, and
 * [ActionDispatcher] does the actual "calling" from plain Kotlin `when` branches. No arbitrary
 * code path the model names ever runs; it only ever picks one of the enumerated
 * [AgentAction.action] values.
 *
 * Tries the real Structured Output API (`generateTypedContentRequest`, genai-prompt 1.0.0-beta4+)
 * first, since when it's available it's strictly better: schema-enforced at the AICore layer,
 * not just prompted for. On this Pixel 10 Pro it currently throws — the SDK classes link fine,
 * but the *on-device* AICore/Nano install hasn't rolled the feature out yet, and the failure is
 * client-side (no logcat trace at all, meaning it never even reaches AICore). Rather than block
 * the whole feature on that rollout, [interpretViaTypedRequest] failing falls back to
 * [interpretViaPromptedJson] — same fail-open pattern as every Nano-backed stage in
 * pixel-nano-assistant: ask for raw JSON matching the schema by prompt alone, then parse it by
 * hand. Less reliable (a misbehaving model could emit invalid JSON) but works everywhere.
 */
class NanoAgent(private val model: GenerativeModel) {

    suspend fun interpret(command: String): AgentAction =
        try {
            interpretViaTypedRequest(command)
        } catch (e: Exception) {
            interpretViaPromptedJson(command)
        }

    private suspend fun interpretViaTypedRequest(command: String): AgentAction {
        val request = GenerateContentRequest.Builder(TextPart(instructionPrompt(command))).build()
        val typedRequest = generateTypedContentRequest(
            generateContentRequest = request,
            outputClass = AgentAction::class
        )
        val response = model.generateContent(typedRequest)
        return response.candidates.firstOrNull()?.response
            ?: error("Model returned no structured response")
    }

    private suspend fun interpretViaPromptedJson(command: String): AgentAction {
        val prompt = instructionPrompt(command) + """


            Respond with ONLY a single JSON object, no markdown fences, no commentary, matching
            exactly this shape (omit fields that don't apply to the chosen action, but always
            include "action" and "confirmation"):
            {"action": "SET_ALARM|SET_TIMER|CREATE_CALENDAR_EVENT|TOGGLE_FLASHLIGHT|OPEN_APP|NONE",
             "hour": <int|null>, "minute": <int|null>, "timerMinutes": <int|null>,
             "eventTitle": <string|null>, "eventStartDateTime": <string|null>,
             "flashlightOn": <bool|null>, "appName": <string|null>, "confirmation": <string>}
        """.trimIndent()

        val response = model.generateContent(prompt)
        val raw = response.candidates.firstOrNull()?.text?.trim()
            ?: error("Model returned no text response")

        // Nano sometimes wraps JSON in ```json fences despite being told not to — strip them
        // rather than fail the whole interpretation over formatting.
        val json = raw.removePrefix("```json").removePrefix("```").removeSuffix("```").trim()
        val obj = JSONObject(json)

        return AgentAction(
            action = obj.getString("action"),
            hour = obj.optIntOrNull("hour"),
            minute = obj.optIntOrNull("minute"),
            timerMinutes = obj.optIntOrNull("timerMinutes"),
            eventTitle = obj.optStringOrNull("eventTitle"),
            eventStartDateTime = obj.optStringOrNull("eventStartDateTime"),
            flashlightOn = obj.optBooleanOrNull("flashlightOn"),
            appName = obj.optStringOrNull("appName"),
            confirmation = obj.optString("confirmation", "")
        )
    }

    private fun instructionPrompt(command: String): String {
        // The model has no wall-clock access on its own, so relative times ("in 20 minutes",
        // "tomorrow at 9am") only resolve correctly if we hand it the current time up front.
        val now = LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME)
        return """
            You are a device-control assistant for an Android phone. The current date and time
            is $now. Given the user's command below, decide which single device action applies
            and fill in only the fields relevant to it. If nothing applies, use action NONE.

            User command: $command
        """.trimIndent()
    }
}

private fun JSONObject.optIntOrNull(name: String): Int? = if (isNull(name) || !has(name)) null else getInt(name)
private fun JSONObject.optStringOrNull(name: String): String? = if (isNull(name) || !has(name)) null else getString(name)
private fun JSONObject.optBooleanOrNull(name: String): Boolean? = if (isNull(name) || !has(name)) null else getBoolean(name)
