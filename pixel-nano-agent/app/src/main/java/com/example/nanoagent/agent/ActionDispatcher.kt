package com.example.nanoagent.agent

import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.provider.AlarmClock
import android.provider.CalendarContract
import java.time.LocalDateTime
import java.time.ZoneId

/**
 * The only code in this app that turns a model-produced [AgentAction] into an actual Android
 * side effect. Every action here either (a) launches another app's own confirmation UI via an
 * implicit intent — deliberately never `EXTRA_SKIP_UI = true` — so a hallucinated or
 * misinterpreted command still requires a human tap before anything is actually created, or
 * (b) is a fully reversible, low-stakes device toggle (flashlight). There is no action here that
 * sends data anywhere, deletes anything, or completes irreversibly without a confirmation screen
 * the user has to look at.
 */
class ActionDispatcher(private val context: Context) {

    sealed interface Result {
        data class Handled(val message: String) : Result
        data class Failed(val reason: String) : Result
    }

    fun dispatch(action: AgentAction): Result = try {
        when (action.action) {
            "SET_ALARM" -> setAlarm(action)
            "SET_TIMER" -> setTimer(action)
            "CREATE_CALENDAR_EVENT" -> createCalendarEvent(action)
            "TOGGLE_FLASHLIGHT" -> toggleFlashlight(action)
            "OPEN_APP" -> openApp(action)
            "NONE" -> Result.Handled(action.confirmation)
            else -> Result.Failed("Model returned an unrecognized action: ${action.action}")
        }
    } catch (e: Exception) {
        Result.Failed("Action failed: ${e.message}")
    }

    private fun setAlarm(action: AgentAction): Result {
        val hour = action.hour ?: return Result.Failed("No hour given for SET_ALARM")
        val minute = action.minute ?: 0
        val intent = Intent(AlarmClock.ACTION_SET_ALARM).apply {
            putExtra(AlarmClock.EXTRA_HOUR, hour)
            putExtra(AlarmClock.EXTRA_MINUTES, minute)
            putExtra(AlarmClock.EXTRA_SKIP_UI, false)
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        if (intent.resolveActivity(context.packageManager) == null) {
            return Result.Failed("No clock app installed to handle SET_ALARM")
        }
        context.startActivity(intent)
        return Result.Handled(action.confirmation)
    }

    private fun setTimer(action: AgentAction): Result {
        val minutes = action.timerMinutes ?: return Result.Failed("No duration given for SET_TIMER")
        val intent = Intent(AlarmClock.ACTION_SET_TIMER).apply {
            putExtra(AlarmClock.EXTRA_LENGTH, minutes * 60)
            putExtra(AlarmClock.EXTRA_SKIP_UI, false)
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        if (intent.resolveActivity(context.packageManager) == null) {
            return Result.Failed("No clock app installed to handle SET_TIMER")
        }
        context.startActivity(intent)
        return Result.Handled(action.confirmation)
    }

    private fun createCalendarEvent(action: AgentAction): Result {
        val title = action.eventTitle ?: return Result.Failed("No title given for CREATE_CALENDAR_EVENT")
        val startIso = action.eventStartDateTime
            ?: return Result.Failed("No start time given for CREATE_CALENDAR_EVENT")

        val startMillis = try {
            LocalDateTime.parse(startIso).atZone(ZoneId.systemDefault()).toInstant().toEpochMilli()
        } catch (e: Exception) {
            return Result.Failed("Couldn't parse event start time \"$startIso\": ${e.message}")
        }

        val intent = Intent(Intent.ACTION_INSERT, CalendarContract.Events.CONTENT_URI).apply {
            putExtra(CalendarContract.Events.TITLE, title)
            putExtra(CalendarContract.EXTRA_EVENT_BEGIN_TIME, startMillis)
            putExtra(CalendarContract.EXTRA_EVENT_END_TIME, startMillis + 60 * 60 * 1000)
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        if (intent.resolveActivity(context.packageManager) == null) {
            return Result.Failed("No calendar app installed to handle CREATE_CALENDAR_EVENT")
        }
        context.startActivity(intent)
        return Result.Handled(action.confirmation)
    }

    private fun toggleFlashlight(action: AgentAction): Result {
        val turnOn = action.flashlightOn ?: return Result.Failed("No on/off value given for TOGGLE_FLASHLIGHT")
        val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val torchCameraId = cameraManager.cameraIdList.firstOrNull { id ->
            cameraManager.getCameraCharacteristics(id)
                .get(CameraCharacteristics.FLASH_INFO_AVAILABLE) == true
        } ?: return Result.Failed("No camera with a flash unit found")

        cameraManager.setTorchMode(torchCameraId, turnOn)
        return Result.Handled(action.confirmation)
    }

    private fun openApp(action: AgentAction): Result {
        val requestedName = action.appName ?: return Result.Failed("No app name given for OPEN_APP")
        val pm = context.packageManager
        val launchable = pm.queryIntentActivities(
            Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER), 0
        )
        val match = launchable.firstOrNull { info ->
            info.loadLabel(pm).toString().contains(requestedName, ignoreCase = true)
        } ?: return Result.Failed("No installed app matching \"$requestedName\"")

        val intent = pm.getLaunchIntentForPackage(match.activityInfo.packageName)
            ?: return Result.Failed("Found \"$requestedName\" but it has no launch intent")
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        context.startActivity(intent)
        return Result.Handled(action.confirmation)
    }
}
