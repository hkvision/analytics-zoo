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
import java.util

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.LoggerFilter
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.models.textclassification.TextClassifier
import com.intel.analytics.zoo.pipeline.api.keras.metrics.Accuracy
import com.intel.analytics.zoo.pipeline.api.keras.objectives.SparseCategoricalCrossEntropy
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.{Pipeline, Transformer}
import org.apache.spark.ml.feature.{SQLTransformer, StopWordsRemover, StringIndexer}
import org.apache.log4j.{Level => Level4j, Logger => Logger4j}
import org.apache.spark.SparkConf
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.types.{ArrayType, DoubleType, StructField, StructType}
import org.slf4j.{Logger, LoggerFactory}
import scopt.OptionParser

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

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

  var classNum: Int = -1

  // Load text, label pairs from file
  def loadRawData(dir: String): ArrayBuffer[(String, Float)] = {
    val texts = ArrayBuffer[String]()
    val labels = ArrayBuffer[Float]()
    // Category is a string name and label is it's one-based index
    val categoryToLabel = new util.HashMap[String, Int]()
    val categoryPathList = new File(dir).listFiles().filter(_.isDirectory).toList.sorted

    categoryPathList.foreach { categoryPath =>
      val label_id = categoryToLabel.size()
      categoryToLabel.put(categoryPath.getName, label_id)
      val textFiles = categoryPath.listFiles()
        .filter(_.isFile).filter(_.getName.forall(Character.isDigit(_))).sorted
      textFiles.foreach { file =>
        val source = Source.fromFile(file, "ISO-8859-1")
        val text = try source.getLines().toList.mkString("\n") finally source.close()
        texts.append(text)
        labels.append(label_id)
      }
    }
    this.classNum = labels.toSet.size
    log.info(s"Found ${texts.length} texts.")
    log.info(s"Found $classNum classes")
    texts.zip(labels)
  }

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
      val spark = SparkSession.builder().config(conf).getOrCreate()
      import spark.implicits._

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

      val data = loadRawData(textDataDir)
      val dataDF = data.toDF("text", "label")
      import org.apache.spark.sql.functions._

      val textDF = dataDF.withColumn("id", monotonically_increasing_id())

      val documentAssembler = new DocumentAssembler().
        setInputCol("text").
        setOutputCol("document")

      val sentenceDetector = new SentenceDetector().
        setInputCols(Array("document")).
        setOutputCol("sentence")

      val regexTokenizer = new Tokenizer().
        setInputCols(Array("sentence")).
        setOutputCol("token")

      val normalizer = new Normalizer().setLowercase(true)
        .setInputCols(Array("token"))
        .setOutputCol("normalized")

      val finisher = new Finisher().
        setInputCols("normalized").
        setCleanAnnotations(false).setOutputCols("finished")

      // input should be array of string
      val remover = new StopWordsRemover()
        .setInputCol("finished")
        .setOutputCol("filtered")

      val pipeline = new Pipeline().
        setStages(Array(
          documentAssembler,
          sentenceDetector,
          regexTokenizer,
          normalizer,
          finisher,
          remover
        ))

      val tokensDF = pipeline
        .fit(textDF)
        .transform(textDF).select("filtered", "label", "id")

      // Transform tokens into indices
      val explodedDF = new SQLTransformer()
        .setStatement("SELECT label, id, explode(filtered) as word FROM __THIS__")
        .transform(tokensDF)

      val stringIndexer = new StringIndexer().setInputCol("word").setOutputCol("index")
      val stringIndexerModel = stringIndexer.fit(explodedDF)
      val labels = stringIndexerModel.labels

      val pipeline2 = new Pipeline().setStages(Array(
        stringIndexer,
        new SQLTransformer()
          .setStatement("""SELECT label, id, index+1 AS index FROM __THIS__ where index<5000.0"""),
        new SQLTransformer()
          .setStatement("""SELECT first(label), id, COLLECT_LIST(index) AS values
                        FROM __THIS__ GROUP BY id"""),
        new Shaper()
      ))
      val shapedIndexedDF = pipeline2.fit(explodedDF)
        .transform(explodedDF)
        .select("first(label, false)", "shaped_values")

      val sampleRDD = shapedIndexedDF.rdd.map(row => {
        Sample(
          featureTensor = Tensor(row.get(1).asInstanceOf[Seq[Double]].toArray.map(_.toFloat),
            Array(sequenceLength)),
          label = row.get(0).asInstanceOf[Float])
      })

      val Array(trainingRDD, valRDD) = sampleRDD.randomSplit(
        Array(trainingSplit, 1 - trainingSplit))

      val model = if (param.model.isDefined) {
        TextClassifier.loadModel(param.model.get)
      }
      else {
        val tokenLength = param.tokenLength
        require(tokenLength == 50 || tokenLength == 100 || tokenLength == 200 || tokenLength == 300,
        s"tokenLength for GloVe can only be 50, 100, 200, 300, but got $tokenLength")
        val wordIndex = (labels.take(5000) zip (Stream from 1)).map(x => x._1 -> x._2).toMap
        val gloveFile = gloveDir + "glove.6B." + tokenLength.toString + "d.txt"
        TextClassifier(classNum, gloveFile, wordIndex, sequenceLength,
          param.encoder, param.encoderOutputDim)
      }

      val optimizer = Optimizer(
        model = model,
        sampleRDD = trainingRDD,
        criterion = SparseCategoricalCrossEntropy[Float](),
        batchSize = param.batchSize
      )

      optimizer
        .setOptimMethod(new Adagrad(learningRate = param.learningRate,
          learningRateDecay = 0.001))
        .setValidation(Trigger.everyEpoch, valRDD, Array(new Accuracy), param.batchSize)
        .setEndWhen(Trigger.maxEpoch(param.nbEpoch))
        .optimize()

      // Predict for probability distributions
      val results = model.predict(valRDD)
      results.take(5)
      // Predict for labels
      val resultClasses = model.predictClasses(valRDD)
      println("First five class predictions (label starts from 0):")
      resultClasses.take(5).foreach(println)

      sc.stop()
    }
  }
}


class Shaper(override val uid: String) extends Transformer {

  def this() = this(Identifiable.randomUID("shaping"))

  def copy(extra: ParamMap): Shaper = {
    defaultCopy(extra)
  }

  override def transformSchema(schema: StructType): StructType = {
    // Check that the input type is a string
    val idx = schema.fieldIndex("values")
    val field = schema.fields(idx)
    require(field.dataType == ArrayType(DoubleType),
      s"Input type must be ArrayType(StringType) but got ${field.dataType}.")
    // Add the return field
    schema.add(StructField("shaped_values", ArrayType(DoubleType), false))
  }

  def transform(df: Dataset[_]): DataFrame = {
    val shaping = udf { in: Seq[Double] => {
      if (in.length > 500) {
        in.slice(in.length - 500, in.length)
      }
      else {
        in ++ Array.fill[Double](500 - in.length)(0)
      }
    } }
    df.select(col("*"),
      shaping(df.col("values")).as("shaped_values"))
  }
}
