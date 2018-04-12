package com.tutsplus.tensorflowtutorial

import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import org.tensorflow.contrib.android.TensorFlowInferenceInterface
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        thread {
            val tfInterface = TensorFlowInferenceInterface(assets,
                    "frozen_model.pb")

            val graph = tfInterface.graph()
            graph.operations().forEach {
                println(it.name())
            }

            tfInterface.feed("my_input/X",
                    floatArrayOf(0f, 1f), 1, 2)

            tfInterface.run(arrayOf("my_output/Sigmoid"))

            val output = floatArrayOf(-1f)
            tfInterface.fetch("my_output/Sigmoid", output)

            println("Output is ${output[0]}")
        }

    }
}
