package com.example.nanoassistant.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.example.nanoassistant.AssistantViewModel

class AssistantViewModelFactory(private val speak: (String) -> Unit) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        @Suppress("UNCHECKED_CAST")
        return AssistantViewModel(speak) as T
    }
}
