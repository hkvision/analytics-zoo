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

package com.intel.analytics.zoo.examples.textclassification

import java.io.File

import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.LoggerFilter
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.text.{TextFeatureToMiniBatch, TextSet}
import com.intel.analytics.zoo.models.textclassification.TextClassifier
import com.intel.analytics.zoo.pipeline.api.keras.metrics.Accuracy
import com.intel.analytics.zoo.pipeline.api.keras.objectives.SparseCategoricalCrossEntropy
import org.apache.log4j.{Level => Level4j, Logger => Logger4j}
import org.apache.spark.SparkConf
import org.slf4j.{Logger, LoggerFactory}
import scopt.OptionParser


case class TextClassificationParams(baseDir: String = "./",
                                    tokenLength: Int = 200,
                                    sequenceLength: Int = 500,
                                    encoder: String = "cnn",
                                    encoderOutputDim: Int = 256,
                                    maxWordsNum: Int = 5000,
                                    trainingSplit: Double = 0.8,
                                    batchSize: Int = 128,
                                    nbEpoch: Int = 20,
                                    learningRate: Double = 0.01,
                                    partitionNum: Int = 4,
                                    model: Option[String] = None)


object TextClassification {
  val log: Logger = LoggerFactory.getLogger(this.getClass)
  LoggerFilter.redirectSparkInfoLogs()
  Logger4j.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level4j.INFO)

  def main(args: Array[String]): Unit = {
    val parser = new OptionParser[TextClassificationParams]("TextClassification Example") {
      opt[String]("baseDir")
        .required()
        .text("The base directory containing the training and word2Vec data")
        .action((x, c) => c.copy(baseDir = x))
      opt[Int]("partitionNum")
        .text("The number of partitions to cut the dataset into")
        .action((x, c) => c.copy(partitionNum = x))
      opt[Int]("tokenLength")
        .text("The size of each word vector, 50 or 100 or 200 or 300 for GloVe")
        .action((x, c) => c.copy(tokenLength = x))
      opt[Int]("sequenceLength")
        .text("The length of a sequence")
        .action((x, c) => c.copy(sequenceLength = x))
      opt[Int]("maxWordsNum")
        .text("The maximum number of words")
        .action((x, c) => c.copy(maxWordsNum = x))
      opt[String]("encoder")
        .text("The encoder for the input sequence, cnn or lstm or gru")
        .action((x, c) => c.copy(encoder = x))
      opt[Int]("encoderOutputDim")
        .text("The output dimension of the encoder")
        .action((x, c) => c.copy(encoderOutputDim = x))
      opt[Double]("trainingSplit")
        .text("The split portion of the data for training")
        .action((x, c) => c.copy(trainingSplit = x))
      opt[Int]('b', "batchSize")
        .text("The number of samples per gradient update")
        .action((x, c) => c.copy(batchSize = x))
      opt[Int]("nbEpoch")
        .text("The number of iterations to train the model")
        .action((x, c) => c.copy(nbEpoch = x))
      opt[Double]('l', "learningRate")
        .text("The learning rate for the TextClassifier model")
        .action((x, c) => c.copy(learningRate = x))
      opt[String]("model")
        .text("Model snapshot location if any")
        .action((x, c) => c.copy(model = Some(x)))
    }

    parser.parse(args, TextClassificationParams()).map { param =>
      val conf = new SparkConf()
        .setAppName("Text Classification Example")
        .set("spark.task.maxFailures", "1")
      val sc = NNContext.initNNContext(conf)

      val sequenceLength = param.sequenceLength
      val trainingSplit = param.trainingSplit
      val textDataDir = s"${param.baseDir}/20news-18828/"
      require(new File(textDataDir).exists(), "Text data directory is not found in baseDir, " +
        "you can run $ANALYTICS_ZOO_HOME/bin/data/news20/get_news20.sh to " +
        "download 20 Newsgroup dataset")
      val gloveDir = s"${param.baseDir}/glove.6B/"
      require(new File(gloveDir).exists(),
        "GloVe word embeddings directory is not found in baseDir, " +
        "you can run $ANALYTICS_ZOO_HOME/bin/data/glove/get_glove.sh to download")

      val textSet = TextSet.read(textDataDir, sc)
      val transformed = textSet.tokenize().normalize()
        .word2idx(removeTopN = 10, maxWordsNum = param.maxWordsNum)
        .shapeSequence(sequenceLength).genSample()

      val Array(trainingRDD, valRDD) = transformed.toDistributed.rdd.randomSplit(
        Array(trainingSplit, 1 - trainingSplit))

      val trainTextSet = DataSet.rdd(trainingRDD) -> TextFeatureToMiniBatch(param.batchSize)
      val valTextSet = DataSet.rdd(valRDD) -> TextFeatureToMiniBatch(param.batchSize)

      val model = if (param.model.isDefined) {
        TextClassifier.loadModel(param.model.get)
      }
      else {
        val tokenLength = param.tokenLength
        require(tokenLength == 50 || tokenLength == 100 || tokenLength == 200 || tokenLength == 300,
        s"tokenLength for GloVe can only be 50, 100, 200, 300, but got $tokenLength")
        val wordIndex = transformed.getWordIndex
        val gloveFile = gloveDir + "glove.6B." + tokenLength.toString + "d.txt"
        TextClassifier(20, gloveFile, wordIndex, sequenceLength,
          param.encoder, param.encoderOutputDim)
      }

      val optimizer = Optimizer(
        model = model,
        dataset = trainTextSet,
        criterion = SparseCategoricalCrossEntropy[Float]()
      )

      optimizer
        .setOptimMethod(new Adagrad(learningRate = param.learningRate,
          learningRateDecay = 0.001))
        .setValidation(Trigger.everyEpoch, valTextSet, Array(new Accuracy))
        .setEndWhen(Trigger.maxEpoch(param.nbEpoch))
        .optimize()

      // TODO: predict on TextSet directly
      // Predict for probability distributions
//      val results = model.predict(valRDD)
//      results.take(5)
      // Predict for labels
//      val resultClasses = model.predictClasses(valRDD)
//      println("First five class predictions (label starts from 0):")
//      resultClasses.take(5).foreach(println)

      sc.stop()
    }
  }
}
