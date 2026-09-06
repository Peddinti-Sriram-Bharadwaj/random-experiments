package com.example.assistant.engine

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Records mic audio and hands back 16kHz mono float PCM — the exact format
 * whisper.cpp's whisper_full() expects. Runs on its own thread per Android's
 * AudioRecord requirements (not a coroutine dispatcher pool thread).
 */
class AudioCapture {
    companion object {
        private const val SAMPLE_RATE = 16_000
    }

    @SuppressLint("MissingPermission") // caller (MainActivity) checks RECORD_AUDIO first
    suspend fun recordUntil(shouldStop: () -> Boolean): FloatArray = withContext(Dispatchers.IO) {
        val minBufSize = AudioRecord.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        val recorder = AudioRecord(
            MediaRecorder.AudioSource.MIC, SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, minBufSize
        )

        val shortBuf = ShortArray(minBufSize)
        val collected = ArrayList<Short>(SAMPLE_RATE * 5) // pre-size ~5s to avoid resizes

        recorder.startRecording()
        try {
            while (!shouldStop()) {
                val read = recorder.read(shortBuf, 0, shortBuf.size)
                for (i in 0 until read) collected.add(shortBuf[i])
            }
        } finally {
            recorder.stop()
            recorder.release()
        }

        // whisper.cpp expects float samples in [-1, 1], not raw 16-bit PCM.
        FloatArray(collected.size) { i -> collected[i] / 32768.0f }
    }
}
