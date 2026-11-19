package com.example.moge3dscanner.ui.main

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.Locale
import kotlin.math.exp

/**
 * High-performance LiteRT / TFLite interpreter for MoGe-3 (and backwards-compatible with MoGe v2).
 * Automatically detects whether the active model is MoGe-3 (CHW 672x672 factorized) or MoGe v2 (HWC 518x518).
 * Supports bundled assets, persistent app-local cache, and automatic first-run download from Hugging Face.
 */
class MogeInterpreter(
    private val context: Context,
    private val onStatusUpdate: ((String) -> Unit)? = null
) {

    companion object {
        private const val TAG = "MogeInterpreter"
        private const val DEFAULT_MODEL_FILENAME = "moge3_dense_stage_weight_only_int8.tflite"
        private const val HF_MODEL_URL =
            "https://huggingface.co/1kaiser/moge3-litert/resolve/main/moge3_dense_stage_weight_only_int8.tflite"
        private const val MIN_VALID_MODEL_BYTES = 10L * 1024L * 1024L // 10 MB sanity floor
    }

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null

    var isMoge3: Boolean = false
        private set

    var isReady: Boolean = false
        private set

    var loadError: String? = null
        private set

    var activeModelName: String = DEFAULT_MODEL_FILENAME
        private set

    var activeAccelerator = "CPU (XNNPACK)"
        private set

    // MoGe Input buffer: 1 * 3 * 518 * 518 * 4 bytes (fits both CHW and HWC)
    private val inputBuffer: ByteBuffer = ByteBuffer.allocateDirect(1 * 3 * 518 * 518 * 4).apply {
        order(ByteOrder.nativeOrder())
    }
    private val inputFloatBuffer: FloatBuffer = inputBuffer.asFloatBuffer()

    // MoGe-3 Output buffers:
    // Output 0: Points [1, 3, 672, 672] (x/z, y/z, logz)
    private val moge3PointsBuffer: ByteBuffer = ByteBuffer.allocateDirect(1 * 3 * 672 * 672 * 4).apply {
        order(ByteOrder.nativeOrder())
    }
    private val moge3PointsFloatBuffer: FloatBuffer = moge3PointsBuffer.asFloatBuffer()

    // Output 1: Normals [1, 3, 672, 672]
    private val moge3NormalsBuffer: ByteBuffer = ByteBuffer.allocateDirect(1 * 3 * 672 * 672 * 4).apply {
        order(ByteOrder.nativeOrder())
    }

    // Output 2: Mask [1, 1, 672, 672]
    private val moge3MaskBuffer: ByteBuffer = ByteBuffer.allocateDirect(1 * 1 * 672 * 672 * 4).apply {
        order(ByteOrder.nativeOrder())
    }

    // Output 3: Metric Scale [1, 1]
    private val moge3ScaleArray = FloatArray(1)

    // Output 4: Features [1, 1024, 42, 42]
    private val moge3FeaturesBuffer: ByteBuffer = ByteBuffer.allocateDirect(1 * 1024 * 42 * 42 * 4).apply {
        order(ByteOrder.nativeOrder())
    }

    // MoGe v2 Output buffer: [1, 518, 518, 3]
    private val mogeV2OutputBuffer: ByteBuffer = ByteBuffer.allocateDirect(1 * 518 * 518 * 3 * 4).apply {
        order(ByteOrder.nativeOrder())
    }
    private val mogeV2OutputFloatBuffer: FloatBuffer = mogeV2OutputBuffer.asFloatBuffer()

    private val pixels518 = IntArray(518 * 518)

    init {
        initModel()
    }

    private fun initModel() {
        val candidates = listOf(
            "moge3_dense_stage_weight_only_int8.tflite",
            "moge3_dense_stage_fp16.tflite",
            "moge_v2_fp16.tflite"
        )

        var loadedModel: String? = null
        var modelBuffer: MappedByteBuffer? = null

        // 1. Check bundled APK assets first
        val assetList = try { context.assets.list("")?.toSet() ?: emptySet() } catch (_: Exception) { emptySet() }
        for (candidate in candidates) {
            if (assetList.contains(candidate)) {
                try {
                    onStatusUpdate?.invoke("Loading bundled $candidate...")
                    modelBuffer = loadAssetModelFile(context, candidate)
                    loadedModel = candidate
                    break
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to load asset candidate $candidate: ${e.message}")
                }
            }
        }

        // 2. Check cached local model files in app storage
        val modelsDir = File(context.noBackupFilesDir, "models").apply { mkdirs() }
        if (modelBuffer == null) {
            for (candidate in candidates) {
                val localFile = File(modelsDir, candidate)
                if (localFile.exists() && localFile.length() >= MIN_VALID_MODEL_BYTES) {
                    try {
                        onStatusUpdate?.invoke("Loading cached 3D AI model...")
                        modelBuffer = mapFileReadOnly(localFile)
                        loadedModel = candidate
                        break
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to map cached model ${localFile.absolutePath}: ${e.message}")
                    }
                }
            }
        }

        // 3. Download MoGe-3 INT8 from Hugging Face if not yet present locally
        if (modelBuffer == null) {
            val targetFile = File(modelsDir, DEFAULT_MODEL_FILENAME)
            try {
                downloadModelFromHuggingFace(HF_MODEL_URL, targetFile)
                onStatusUpdate?.invoke("Initializing 3D AI engine...")
                modelBuffer = mapFileReadOnly(targetFile)
                loadedModel = DEFAULT_MODEL_FILENAME
            } catch (e: Exception) {
                loadError = e.message ?: "Download failed"
                Log.e(TAG, "Failed to download MoGe model from Hugging Face", e)
                return
            }
        }

        if (modelBuffer == null) {
            loadError = "No suitable MoGe model available"
            Log.e(TAG, "No suitable MoGe model found!")
            return
        }

        activeModelName = loadedModel ?: DEFAULT_MODEL_FILENAME

        // Attempt GPU acceleration first, fall back to CPU (XNNPACK)
        try {
            val gpuOptions = Interpreter.Options().apply {
                gpuDelegate = GpuDelegate()
                addDelegate(gpuDelegate)
            }
            interpreter = Interpreter(modelBuffer, gpuOptions)
            activeAccelerator = "GPU"
            Log.i(TAG, "Successfully initialized TFLite ($activeModelName) with GPU Delegate.")
        } catch (gpuException: Throwable) {
            Log.w(TAG, "GPU Delegate unavailable. Falling back to CPU.", gpuException)
            try {
                gpuDelegate?.close()
            } catch (_: Throwable) {}
            gpuDelegate = null

            try {
                val cpuOptions = Interpreter.Options().apply {
                    setNumThreads(4)
                }
                interpreter = Interpreter(modelBuffer, cpuOptions)
                activeAccelerator = "CPU (XNNPACK)"
                Log.i(TAG, "Successfully initialized TFLite ($activeModelName) with CPU.")
            } catch (cpuException: Throwable) {
                loadError = cpuException.message ?: "TFLite initialization failed"
                Log.e(TAG, "CPU TFLite initialization failed", cpuException)
                return
            }
        }

        // Determine architecture from input tensor layout:
        val interp = interpreter
        if (interp != null) {
            val inShape = interp.getInputTensor(0).shape()
            // MoGe-3 has shape [1, 3, 518, 518] (CHW), whereas MoGe v2 has [1, 518, 518, 3] (HWC)
            isMoge3 = (inShape.size == 4 && inShape[1] == 3)
            isReady = true
            loadError = null
            Log.i(TAG, "Architecture detected: isMoge3=$isMoge3 (Input Shape: ${inShape.joinToString()})")
        }
    }

    private fun downloadModelFromHuggingFace(sourceUrl: String, targetFile: File) {
        val partFile = File(targetFile.parentFile, "${targetFile.name}.part")
        if (partFile.exists()) {
            partFile.delete()
        }

        onStatusUpdate?.invoke("Connecting to Hugging Face for MoGe-3 model...")
        var currentUrl = URL(sourceUrl)
        var connection: HttpURLConnection? = null
        var redirects = 0

        while (redirects < 6) {
            val conn = (currentUrl.openConnection() as HttpURLConnection).apply {
                instanceFollowRedirects = false
                connectTimeout = 20_000
                readTimeout = 60_000
                setRequestProperty("User-Agent", "VisLocMoGe3DScanner/3.0 Android")
            }
            val code = conn.responseCode
            if (code in listOf(301, 302, 303, 307, 308)) {
                val location = conn.getHeaderField("Location")
                    ?: throw IllegalStateException("Redirect $code missing Location header")
                conn.disconnect()
                currentUrl = URL(currentUrl, location)
                redirects++
            } else {
                connection = conn
                break
            }
        }

        val activeConn = connection ?: throw IllegalStateException("Too many redirects downloading model")
        try {
            val responseCode = activeConn.responseCode
            if (responseCode != HttpURLConnection.HTTP_OK) {
                throw IllegalStateException("HTTP $responseCode while downloading model")
            }

            val totalBytes = activeConn.contentLengthLong
            var downloadedBytes = 0L
            var lastReportTime = System.currentTimeMillis()
            var lastReportedBytes = 0L

            activeConn.inputStream.buffered(64 * 1024).use { input ->
                FileOutputStream(partFile).use { output ->
                    val buffer = ByteArray(64 * 1024)
                    while (true) {
                        val read = input.read(buffer)
                        if (read == -1) break
                        output.write(buffer, 0, read)
                        downloadedBytes += read

                        val now = System.currentTimeMillis()
                        val elapsedSec = (now - lastReportTime) / 1000.0
                        if (elapsedSec >= 0.35 || (totalBytes > 0 && downloadedBytes == totalBytes)) {
                            val mbLoaded = downloadedBytes / (1024.0 * 1024.0)
                            val speedMb = if (elapsedSec > 0) {
                                ((downloadedBytes - lastReportedBytes) / (1024.0 * 1024.0)) / elapsedSec
                            } else 0.0
                            val msg = if (totalBytes > 0) {
                                val pct = (downloadedBytes * 100.0 / totalBytes).coerceIn(0.0, 100.0)
                                val mbTotal = totalBytes / (1024.0 * 1024.0)
                                String.format(
                                    Locale.US,
                                    "Downloading MoGe-3: %.0f%% (%.0f/%.0f MB • %.1f MB/s)",
                                    pct, mbLoaded, mbTotal, speedMb
                                )
                            } else {
                                String.format(
                                    Locale.US,
                                    "Downloading MoGe-3: %.1f MB (%.1f MB/s)",
                                    mbLoaded, speedMb
                                )
                            }
                            onStatusUpdate?.invoke(msg)
                            lastReportTime = now
                            lastReportedBytes = downloadedBytes
                        }
                    }
                    output.flush()
                }
            }

            if (partFile.length() < MIN_VALID_MODEL_BYTES) {
                partFile.delete()
                throw IllegalStateException("Downloaded model file incomplete (${partFile.length()} bytes)")
            }

            if (targetFile.exists()) {
                targetFile.delete()
            }
            if (!partFile.renameTo(targetFile)) {
                partFile.copyTo(targetFile, overwrite = true)
                partFile.delete()
            }
            Log.i(TAG, "Downloaded MoGe-3 model to ${targetFile.absolutePath} (${targetFile.length()} bytes)")
        } finally {
            activeConn.disconnect()
        }
    }

    private fun loadAssetModelFile(context: Context, modelName: String): MappedByteBuffer {
        context.assets.openFd(modelName).use { fileDescriptor ->
            FileInputStream(fileDescriptor.fileDescriptor).use { inputStream ->
                val fileChannel = inputStream.channel
                return fileChannel.map(
                    FileChannel.MapMode.READ_ONLY,
                    fileDescriptor.startOffset,
                    fileDescriptor.declaredLength
                )
            }
        }
    }

    private fun mapFileReadOnly(file: File): MappedByteBuffer {
        FileInputStream(file).use { inputStream ->
            return inputStream.channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
        }
    }

    @Synchronized
    fun runInference(bitmap: Bitmap, stride: Int): Pair<FloatArray, FloatArray>? =
        runInferenceWithColor(bitmap, bitmap, stride)

    @Synchronized
    fun runInferenceWithColor(depthBitmap: Bitmap, colorBitmap: Bitmap, stride: Int): Pair<FloatArray, FloatArray>? {
        val interp = interpreter ?: return null

        val scaledDepth = Bitmap.createScaledBitmap(depthBitmap, 518, 518, true)
        scaledDepth.getPixels(pixels518, 0, 518, 0, 0, 518, 518)

        inputFloatBuffer.rewind()

        if (isMoge3) {
            // Planar format: CHW [1, 3, 518, 518]
            val totalPixels = 518 * 518
            // R channel
            for (i in 0 until totalPixels) {
                inputFloatBuffer.put(((pixels518[i] shr 16) and 0xFF) / 255.0f)
            }
            // G channel
            for (i in 0 until totalPixels) {
                inputFloatBuffer.put(((pixels518[i] shr 8) and 0xFF) / 255.0f)
            }
            // B channel
            for (i in 0 until totalPixels) {
                inputFloatBuffer.put((pixels518[i] and 0xFF) / 255.0f)
            }

            // Outputs for MoGe-3
            val inputs = arrayOf<Any>(inputBuffer)
            val outputs = mutableMapOf<Int, Any>()
            outputs[0] = moge3PointsBuffer
            outputs[1] = moge3NormalsBuffer
            outputs[2] = moge3MaskBuffer
            outputs[3] = moge3ScaleArray
            outputs[4] = moge3FeaturesBuffer

            inputBuffer.rewind()
            moge3PointsBuffer.rewind()
            moge3PointsFloatBuffer.rewind()

            interp.runForMultipleInputsOutputs(inputs, outputs)

            val metricScale = exp(moge3ScaleArray[0].toDouble()).toFloat()

            // Output resolution is 672 x 672
            val outW = 672
            val outH = 672
            val planeSize = outW * outH

            val scaledColor = Bitmap.createScaledBitmap(colorBitmap, outW, outH, true)
            val colorPixels = IntArray(planeSize)
            scaledColor.getPixels(colorPixels, 0, outW, 0, 0, outW, outH)

            val step = stride
            val stepsX = (outW + step - 1) / step
            val size = stepsX * stepsX
            val positions = FloatArray(size * 3)
            val colors = FloatArray(size * 3)

            var idx = 0
            for (y in 0 until outH step step) {
                for (x in 0 until outW step step) {
                    val pixelIdx = y * outW + x
                    // CHW planar layout:
                    val xOverZ = moge3PointsFloatBuffer.get(pixelIdx)
                    val yOverZ = moge3PointsFloatBuffer.get(planeSize + pixelIdx)
                    val logZ = moge3PointsFloatBuffer.get(planeSize * 2 + pixelIdx)

                    val z = exp(logZ.toDouble()).toFloat() * metricScale
                    val px = xOverZ * z
                    val py = yOverZ * z

                    positions[idx * 3]     = px
                    positions[idx * 3 + 1] = py
                    positions[idx * 3 + 2] = z

                    val colorP = colorPixels[pixelIdx]
                    colors[idx * 3]     = ((colorP shr 16) and 0xFF) / 255.0f
                    colors[idx * 3 + 1] = ((colorP shr 8)  and 0xFF) / 255.0f
                    colors[idx * 3 + 2] = (colorP and 0xFF) / 255.0f
                    idx++
                }
            }

            return Pair(positions, colors)

        } else {
            // Interleaved format: HWC [1, 518, 518, 3] (MoGe v2)
            for (i in pixels518.indices) {
                val pixel = pixels518[i]
                inputFloatBuffer.put(((pixel shr 16) and 0xFF) / 255.0f)
                inputFloatBuffer.put(((pixel shr 8) and 0xFF) / 255.0f)
                inputFloatBuffer.put((pixel and 0xFF) / 255.0f)
            }

            val inputs = arrayOf<Any>(inputBuffer)
            val outputs = mutableMapOf<Int, Any>()
            val scaleArr = FloatArray(1)
            outputs[0] = scaleArr
            outputs[1] = mogeV2OutputBuffer

            inputBuffer.rewind()
            mogeV2OutputBuffer.rewind()
            mogeV2OutputFloatBuffer.rewind()
            interp.runForMultipleInputsOutputs(inputs, outputs)

            val scale = if (scaleArr[0] > 0f) scaleArr[0] else 1.0f

            val scaledColor = Bitmap.createScaledBitmap(colorBitmap, 518, 518, true)
            val colorPixels = IntArray(518 * 518)
            scaledColor.getPixels(colorPixels, 0, 518, 0, 0, 518, 518)

            val step = stride
            val stepsX = (518 + step - 1) / step
            val size = stepsX * stepsX
            val positions = FloatArray(size * 3)
            val colors = FloatArray(size * 3)

            var idx = 0
            for (y in 0 until 518 step step) {
                for (x in 0 until 518 step step) {
                    val i = y * 518 + x
                    positions[idx * 3]     = mogeV2OutputFloatBuffer.get(i * 3) * scale
                    positions[idx * 3 + 1] = mogeV2OutputFloatBuffer.get(i * 3 + 1) * scale
                    positions[idx * 3 + 2] = mogeV2OutputFloatBuffer.get(i * 3 + 2) * scale

                    val pixel = colorPixels[i]
                    colors[idx * 3]     = ((pixel shr 16) and 0xFF) / 255.0f
                    colors[idx * 3 + 1] = ((pixel shr 8)  and 0xFF) / 255.0f
                    colors[idx * 3 + 2] = (pixel and 0xFF) / 255.0f
                    idx++
                }
            }

            return Pair(positions, colors)
        }
    }

    fun sampleColors(bitmap: Bitmap, stride: Int = 4): FloatArray {
        val outDim = if (isMoge3) 672 else 518
        val scaled = Bitmap.createScaledBitmap(bitmap, outDim, outDim, true)
        val pixels = IntArray(outDim * outDim)
        scaled.getPixels(pixels, 0, outDim, 0, 0, outDim, outDim)

        val step = stride
        val stepsX = (outDim + step - 1) / step
        val size = stepsX * stepsX
        val colors = FloatArray(size * 3)

        var idx = 0
        for (y in 0 until outDim step step) {
            for (x in 0 until outDim step step) {
                val i = y * outDim + x
                val pixel = pixels[i]
                colors[idx * 3]     = ((pixel shr 16) and 0xFF) / 255.0f
                colors[idx * 3 + 1] = ((pixel shr 8)  and 0xFF) / 255.0f
                colors[idx * 3 + 2] = (pixel and 0xFF) / 255.0f
                idx++
            }
        }
        return colors
    }

    fun close() {
        try {
            interpreter?.close()
        } catch (_: Throwable) {}
        try {
            gpuDelegate?.close()
        } catch (_: Throwable) {}
    }
}
