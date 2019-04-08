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

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.log4j.Logger
import scopt.OptionParser


case class ResNet50PerfParams(model: String = "",
                              batchSize: Int = 32,
                              iteration: Int = 2000)

object Perf {

  val logger: Logger = Logger.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    System.setProperty("bigdl.localMode", "true")
    System.setProperty("bigdl.engineType", "mkldnn")

    val parser = new OptionParser[ResNet50PerfParams]("Int8 Performance Test") {
      opt[String]('m', "model")
        .text("The path to the int8 quantized ResNet50 model snapshot")
        .action((v, p) => p.copy(model = v))
        .required()
      opt[Int]('b', "batchSize")
        .text("Batch size of input data")
        .action((v, p) => p.copy(batchSize = v))
      opt[Int]('i', "iteration")
        .text("Iteration of perf test. The result will be average of each iteration time cost")
        .action((v, p) => p.copy(iteration = v))
    }

    parser.parse(args, ResNet50PerfParams()).foreach { params =>
      val batchSize = params.batchSize
      val inputShape = Array(batchSize, 3, 224, 224)
      val input = Tensor(inputShape).rand()

      // fp32 model running on MKLDNN
//      val fp32model = Module.loadModule[Float](params.model) // .quantize()
//      val model = ConversionUtils.convert[Float](fp32model)
//      model.evaluate()

      val model = Module.loadModule[Float](params.model).quantize()

      var iteration = 0
      while (iteration < params.iteration) {
        val start = System.nanoTime()
        model.forward(input)
        val timeUsed = System.nanoTime() - start

        val throughput = "%.2f".format(batchSize.toFloat / (timeUsed / 1e9))
        logger.info(s"Iteration $iteration, takes $timeUsed ns, throughput is $throughput imgs/sec")

        iteration += 1
      }
    }
  }
}