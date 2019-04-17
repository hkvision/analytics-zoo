/*
 * Copyright 2018 Analytics Zoo Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.zoo.examples.vnni.bigdl

import com.intel.analytics.bigdl.dataset.image.CropCenter
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.feature.image._
import com.intel.analytics.zoo.models.image.imageclassification.{ImageClassifier, LabelOutput}
import org.apache.log4j.Logger
import scopt.OptionParser


case class PredictLatencyParams(model: String = "",
                                iteration: Int = 1000,
                                quantize: Boolean = true,
                                folder: String = "./")

object PredictSingle {

  val logger: Logger = Logger.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    System.setProperty("bigdl.localMode", "true")
    System.setProperty("bigdl.engineType", "mkldnn")

    val parser = new OptionParser[PredictLatencyParams]("Latency Performance Test") {
      opt[String]('m', "model")
        .text("The path to the int8 quantized ResNet50 model snapshot")
        .action((v, p) => p.copy(model = v))
        .required()
      opt[Boolean]("quantize")
        .action((v, p) => p.copy(quantize = v))
      opt[Int]('i', "iteration")
        .text("Iteration of perf test. The result will be average of each iteration time cost")
        .action((v, p) => p.copy(iteration = v))
      opt[String]('f', "folder")
        .text("The path to the image data")
        .action((x, c) => c.copy(folder = x))
        .required()
    }

    parser.parse(args, PredictLatencyParams()).foreach { param =>
      Engine.init

      val model = ImageClassifier.loadModel[Float](param.model, quantize = param.quantize)
      val preprocessor = ImageRandomResize(256, 256) ->
        ImageRandomCropper(224, 224, false, CropCenter) ->
        ImageChannelScaledNormalizer(104, 117, 123, 0.0078125) ->
        ImageMatToTensor()
      model.setEvaluateStatus()

      var iteration = 0
      while (iteration < param.iteration) {
        val start = System.nanoTime()
        val start1 = System.nanoTime()
        val images = ImageSet.read(param.folder).toLocal()
        val latency1 = System.nanoTime() - start1
        logger.info(s"Iteration $iteration, read image latency is ${latency1 / 1e6} ms")
        val start2 = System.nanoTime()
        val preprocessed = images -> preprocessor
        val latency2 = System.nanoTime() - start2
        logger.info(s"Iteration $iteration, preprocessing latency is ${latency2 / 1e6} ms")
        val image = preprocessed.toLocal().array.head
        val input = image[Tensor[Float]](ImageFeature.imageTensor).resize(1, 3, 224, 224)
        val start3 = System.nanoTime()
        val output = model.forward(input)
        val latency3 = System.nanoTime() - start3
        logger.info(s"Iteration $iteration, forward latency is ${latency3 / 1e6} ms")
        image(ImageFeature.predict) = output
        val start4 = System.nanoTime()
        val labelOutput = LabelOutput(model.getConfig().labelMap, "clses",
          "probs", probAsInput = false)
        val prediction = labelOutput.transform(image)
        val latency4 = System.nanoTime() - start4
        logger.info(s"Iteration $iteration, postprocessing latency is ${latency4 / 1e6} ms")
        val latency = System.nanoTime() - start
        logger.info(s"Iteration $iteration, latency is ${latency / 1e6} ms")
        logger.info(s"------------------------------------------------")

        iteration += 1
      }
    }
  }
}
