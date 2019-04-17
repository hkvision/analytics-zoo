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

import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.image.ImageSet
import com.intel.analytics.zoo.models.image.imageclassification.{ImageClassifier, LabelOutput}
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser

case class ImageClassificationParams(folder: String = "./",
                                     model: String = "",
                                     topN: Int = 5,
                                     partitionNum: Int = 4,
                                     quantize: Boolean = true)

object Predict {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  val logger: Logger = Logger.getLogger(getClass)

  def main(args: Array[String]): Unit = {
//    System.setProperty("bigdl.engineType", "mkldnn")
    val parser = new OptionParser[ImageClassificationParams]("ResNet50 Int8 Inference Example") {
      opt[String]('f', "folder")
        .text("The path to the image data")
        .action((x, c) => c.copy(folder = x))
        .required()
      opt[String]('m', "model")
        .text("The path to the int8 quantized ResNet50 model snapshot")
        .action((x, c) => c.copy(model = x))
        .required()
      opt[Int]("topN")
        .text("top N number")
        .action((x, c) => c.copy(topN = x))
      opt[Int]("partitionNum")
        .text("The number of partitions to cut the dataset into")
        .action((x, c) => c.copy(partitionNum = x))
      opt[Boolean]("quantize")
        .action((v, p) => p.copy(quantize = v))
    }
    parser.parse(args, ImageClassificationParams()).map(param => {
      val start = System.nanoTime()
      val sc = NNContext.initNNContext("ResNet50 Int8 Inference Example")
      val start1 = System.nanoTime()
      val images = ImageSet.read(param.folder)
      val duration1 = System.nanoTime() - start1
      logger.info(s"Read images time: ${duration1 / 1e9} seconds")

      val start2 = System.nanoTime()
      val model = ImageClassifier.loadModel[Float](param.model, quantize = param.quantize)
      val duration2 = System.nanoTime() - start2
      logger.info(s"Load model time: ${duration2 / 1e9} seconds")
      val output = model.predictImageSet(images)

      val start3 = System.nanoTime()
      val labelOutput = LabelOutput(model.getConfig().labelMap, "clses",
        "probs", probAsInput = false)
      val result = labelOutput(output).toLocal().array
      val duration3 = System.nanoTime() - start3
      logger.info(s"Postprocessing time: ${duration3 / 1e9} seconds")

      val duration = System.nanoTime() - start
      logger.info(s"Total time: ${duration / 1e9} seconds")

      sc.stop()
    })
  }
}
