package com.mobilecomputing.mobilecomputingproject

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import com.google.gson.Gson
import com.mobilecomputing.mobilecomputingproject.databinding.ActivityQuizBinding
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.schema.ResizeBilinearOptions
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStreamReader
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*

object BitmapHolder {
    var bitmap: Bitmap? = null
}

class QuizActivity : AppCompatActivity() {

    private lateinit var viewBinding: ActivityQuizBinding

    private lateinit var encoderDNN: Interpreter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        viewBinding = ActivityQuizBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        if (BitmapHolder.bitmap != null) {
            viewBinding.quizImageViewCaptured.setImageBitmap(BitmapHolder.bitmap)
        }

        viewBinding.quizCaptureImageButton.setOnClickListener {
            val intent = Intent(this, CameraPreviewActivity::class.java)
            startActivity(intent)
        }

        viewBinding.quizBackToHomeButton.setOnClickListener {
            BitmapHolder.bitmap = null
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
        }

        viewBinding.showAnswerButton.setOnClickListener {
            viewBinding.quizAnswerTextview.visibility = View.VISIBLE
        }

        viewBinding.hideAnswerButton.setOnClickListener {
            viewBinding.quizAnswerTextview.visibility = View.INVISIBLE
        }

        try {
            val tfLiteModel = loadDNNModelFile("quantized_encoder.tflite")
            encoderDNN = Interpreter(tfLiteModel, Interpreter.Options())
            Log.d("Encoder DNN", "Print encoder shape")
            printTensorShape(encoderDNN)
        } catch (e: Exception) {
            Log.d("Encoder DNN", "tensorflow lite model not loaded")
            e.printStackTrace()
        }

        if (BitmapHolder.bitmap == null) {
            val stream = this.assets.open("surf.jpeg")
            var bitmap = BitmapFactory.decodeStream(stream)
            runCaptionInference(bitmap)
        } else {
            runCaptionInference(BitmapHolder.bitmap!!)
        }
    }

    override fun onDestroy() {
        viewBinding.quizAnswerTextview.visibility = View.INVISIBLE
        viewBinding.quizAnswerTextview.text = "Processing image ..."
        super.onDestroy()
    }

    private fun runCaptionInference(bitmap: Bitmap) {
        val resizedTensorImage = getResizedTensorImage(bitmap)
        viewBinding.quizImageViewCaptured.setImageBitmap(resizedTensorImage.bitmap)
        Log.d("Tensor Image", "Type = ${resizedTensorImage.dataType}, Shape = (${resizedTensorImage.height}, ${resizedTensorImage.width})")
        val encoderOutput = runEncoder(resizedTensorImage)
        runDecoder(encoderOutput)
    }

    private fun getResizedTensorImage(bitmap: Bitmap): TensorImage {
        val tensorImage = TensorImage.createFrom(TensorImage.fromBitmap(bitmap), DataType.FLOAT32)
        ResizeBilinearOptions()
        val resizeOp = ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR)
        return resizeOp.apply(tensorImage)
    }

    private fun runDecoder(encoderOutput: Array<Array<Array<FloatArray>>>) {
        val gson = Gson()
        val jsonData = gson.toJson(encoderOutput)
        val client = OkHttpClient()
        val requestBody = jsonData.toRequestBody("application/json; charset=utf-8".toMediaTypeOrNull())

        val request = Request.Builder()
            .url("http://192.168.0.6:8123/decode") // Modify this IP address if necessary
            .post(requestBody)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                e.printStackTrace()
                Log.d("HTTP REQUEST", "Request fail")
            }

            override fun onResponse(call: Call, response: Response) {
                val decoderOutput = response.body?.string()
                runOnUiThread {
                    viewBinding.quizAnswerTextview.text = decoderOutput
                }
                Log.d("HTTP REQUEST", "Request success :) Response : $decoderOutput")
            }
        })
    }

    private fun runEncoder(resizedTensorImage: TensorImage): Array<Array<Array<FloatArray>>> {
        val encoderInput = resizedTensorImage.buffer
        val encoderOutput = Array(1){Array(7){Array(7){FloatArray(2048) } } }
        encoderDNN.run(encoderInput, encoderOutput)
        return encoderOutput
    }

    private fun printTensorShape(dnn: Interpreter) {
        Log.d("INPUT TENSOR SHAPE", "Input tensor count = ${dnn.inputTensorCount}")
        for (i in 0 until dnn.inputTensorCount) {
            val inputTensor = dnn.getInputTensor(i)
            Log.d("INPUT TENSOR SHAPE", "${i}th input tensor shape = ${Arrays.toString(inputTensor.shape())}")
            Log.d("INPUT TENSOR TYPE", "${i}th input tensor type = ${inputTensor.dataType()}")
        }
        Log.d("OUTPUT TENSOR SHAPE", "Output tensor count = ${dnn.outputTensorCount}")
        for (i in 0 until dnn.outputTensorCount) {
            val outputTensor = dnn.getOutputTensor(i)
            Log.d("OUTPUT TENSOR SHAPE", "${i}th output tensor shape = ${Arrays.toString(outputTensor.shape())}")
            Log.d("OUTPUT TENSOR TYPE", "${i}th output tensor type = ${outputTensor.dataType()}")
        }
    }

    private fun loadDNNModelFile(filename: String): MappedByteBuffer {
        val fd = this.assets.openFd(filename)
        val fileChannel = FileInputStream(fd.fileDescriptor).channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }
}