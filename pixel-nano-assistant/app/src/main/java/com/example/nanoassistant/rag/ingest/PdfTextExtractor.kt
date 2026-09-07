package com.example.nanoassistant.rag.ingest

import android.content.Context
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.text.PDFTextStripper
import java.io.InputStream

/** Extracts plain text from a PDF via PDFBox-Android (pure-Java PDFBox port, no native code).
 *  Real-world PDF text comes out messier than authored plaintext: broken ligatures, page
 *  headers/footers interleaved with body text, metadata tables with labels and values on
 *  separate lines — useful for stress-testing chunking on non-curated documents. */
object PdfTextExtractor {

    private var initialized = false

    fun extractText(context: Context, input: InputStream): String {
        if (!initialized) {
            PDFBoxResourceLoader.init(context.applicationContext)
            initialized = true
        }
        PDDocument.load(input).use { document ->
            return PDFTextStripper().getText(document)
        }
    }
}
