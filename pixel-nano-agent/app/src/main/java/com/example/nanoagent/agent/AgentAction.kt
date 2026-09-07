package com.example.nanoagent.agent

import com.google.mlkit.genai.schema.annotations.Generable
import com.google.mlkit.genai.schema.annotations.Guide

/**
 * The "function call" AICore's Structured Output API stands in for. Gemini Nano via AICore has
 * no native FunctionDeclaration/Tool object (see [NanoAgent] doc comment) — so instead of the
 * model choosing from a set of declared tools, it fills in this single discriminated-union-style
 * object, and [com.example.nanoagent.agent.ActionDispatcher] is the only code that ever turns
 * [action] into a real Android side effect. Every field the model can steer is enumerated here;
 * there's no way for the model to reach any Android API this class doesn't expose a field for.
 */
@Generable(description = "The single device action to take in response to a user's command")
data class AgentAction(
    @Guide(
        description = "Which action to take. Use NONE if the command doesn't map to a device " +
            "action — just answer conversationally in the confirmation field instead.",
        enumValues = ["SET_ALARM", "SET_TIMER", "CREATE_CALENDAR_EVENT", "TOGGLE_FLASHLIGHT", "OPEN_APP", "NONE"]
    )
    val action: String,

    @Guide(description = "For SET_ALARM: the hour in 24-hour format (0-23).")
    val hour: Int?,

    @Guide(description = "For SET_ALARM: the minute (0-59).")
    val minute: Int?,

    @Guide(description = "For SET_TIMER: the timer duration in minutes.")
    val timerMinutes: Int?,

    @Guide(description = "For CREATE_CALENDAR_EVENT: a short title for the event.")
    val eventTitle: String?,

    @Guide(
        description = "For CREATE_CALENDAR_EVENT: the event start time as an ISO-8601 local " +
            "date-time, e.g. 2026-09-07T15:30:00. Resolve relative times (\"tomorrow at 9am\") " +
            "against the current time given in the prompt."
    )
    val eventStartDateTime: String?,

    @Guide(description = "For TOGGLE_FLASHLIGHT: true to turn it on, false to turn it off.")
    val flashlightOn: Boolean?,

    @Guide(description = "For OPEN_APP: the display name of the app to open, e.g. \"Camera\".")
    val appName: String?,

    @Guide(description = "A short, natural-language confirmation of what you're about to do (or, for NONE, a direct conversational reply).")
    val confirmation: String
)
