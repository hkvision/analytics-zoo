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

package com.intel.analytics.zoo.models.textclassification

import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl.Criterion
import com.intel.analytics.bigdl.dataset.{DistributedDataSet, Sample, SampleToMiniBatch, Utils}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim.{OptimMethod, ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{RandomGenerator, Shape}
import com.intel.analytics.zoo.feature.text.TextSet
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import com.intel.analytics.zoo.pipeline.api.keras.models.{KerasNet, Sequential}
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
 * The model used for text classification.
 */
class TextClassifier[T: ClassTag] private(
    val classNum: Int,
    val tokenLength: Int,
    val sequenceLength: Int = 500,
    val encoder: String = "cnn",
    val encoderOutputDim: Int = 256,
    val embedding: Embedding[T] = null)(implicit ev: TensorNumeric[T])
  extends ZooModel[Activity, Activity, T] {

  override def buildModel(): AbstractModule[Activity, Activity, T] = {
    val model = Sequential[T]()
    if (embedding != null) {
      model.add(embedding)
    }
    else {
      model.add(InputLayer(inputShape = Shape(sequenceLength, tokenLength)))
    }
    if (encoder.toLowerCase() == "cnn") {
      model.add(Convolution1D(encoderOutputDim, 5, activation = "relu"))
      model.add(GlobalMaxPooling1D())
    }
    else if (encoder.toLowerCase() == "lstm") {
      model.add(LSTM(encoderOutputDim))
    }
    else if (encoder.toLowerCase() == "gru") {
      model.add(GRU(encoderOutputDim))
    }
    else {
      throw new IllegalArgumentException(s"Unsupported encoder for TextClassifier: $encoder")
    }
    model.add(Dense(128))
    model.add(Dropout(0.2))
    model.add(Activation("relu"))
    model.add(Dense(classNum, activation = "softmax"))
    model
  }

  // For the following methods, please refer to KerasNet for documentation.
  def compile(
      optimizer: OptimMethod[T],
      loss: Criterion[T],
      metrics: List[ValidationMethod[T]] = null)(implicit ev: TensorNumeric[T]): Unit = {
    model.asInstanceOf[KerasNet[T]].compile(optimizer, loss, metrics)
  }

  def fit(
      x: TextSet,
      batchSize: Int,
      nbEpoch: Int,
      validationData: TextSet = null)(implicit ev: TensorNumeric[T]): Unit = {
    val sampleRDD = x.toDistributed().rdd.map(_.getSample).asInstanceOf[RDD[Sample[T]]]
    val nodeNumber = EngineRef.getNodeNumber()
    val dataset = new CachedDistriDataSet[Sample[T]](
      sampleRDD.coalesce(nodeNumber, true)
        .mapPartitions(iter => {
          Iterator.single(iter.toArray)
        }).setName("cached dataset")
        .cache()
    ) -> SampleToMiniBatch[T](batchSize)
    model.asInstanceOf[KerasNet[T]].fit(dataset, nbEpoch)
  }

  def evaluate(
      x: TextSet,
      batchSize: Int)
    (implicit ev: TensorNumeric[T]): Array[(ValidationResult, ValidationMethod[T])] = {
    model.asInstanceOf[KerasNet[T]].evaluate(x, batchSize)
  }

  def predict(
      x: TextSet,
      batchPerThread: Int): TextSet = {
    model.asInstanceOf[KerasNet[T]].predict(x, batchPerThread)
  }

  def setTensorBoard(logDir: String, appName: String): Unit = {
    model.asInstanceOf[KerasNet[T]].setTensorBoard(logDir, appName)
  }

  def setCheckpoint(path: String, overWrite: Boolean = true): Unit = {
    model.asInstanceOf[KerasNet[T]].setCheckpoint(path, overWrite)
  }
}


class CachedDistriDataSet[T: ClassTag]
(buffer: RDD[Array[T]], isInOrder: Boolean = false, groupSize: Int = 1)
  extends DistributedDataSet[T] {

  protected lazy val count: Long = buffer.mapPartitions(iter => {
    require(iter.hasNext)
    val array = iter.next()
    require(!iter.hasNext)
    Iterator.single(array.length)
  }).reduce(_ + _)

  protected var indexes: RDD[Array[Int]] = buffer.mapPartitions(iter => {
    Iterator.single((0 until iter.next().length).toArray)
  }).setName("original index").cache()

  override def data(train: Boolean): RDD[T] = {
    val _train = train
    val _groupSize = if (isInOrder) Utils.getBatchSize(groupSize) else 1
    buffer.zipPartitions(indexes)((dataIter, indexIter) => {
      val indexes = indexIter.next()
      val indexOffset = math.max(1, indexes.length - (_groupSize - 1))
      val localData = dataIter.next()
      val offset = if (_train) {
        RandomGenerator2.RNG.uniform(0, indexOffset).toInt
      } else {
        0
      }
      new Iterator[T] {
        private val _offset = new AtomicInteger(offset)

        override def hasNext: Boolean = {
          if (_train) true else _offset.get() < localData.length
        }

        override def next(): T = {
          val i = _offset.getAndIncrement()
          if (_train) {
            localData(indexes(i % localData.length))
          } else {
            if (i < localData.length) {
              localData(indexes(i))
            } else {
              null.asInstanceOf[T]
            }
          }
        }
      }
    })
  }

  override def size(): Long = count

  override def shuffle(): Unit = {
    if (!isInOrder) {
      indexes.unpersist()
      indexes = buffer.mapPartitions(iter => {
        Iterator.single(RandomGenerator2.shuffle((0 until iter.next().length).toArray))
      }).setName("shuffled index").cache()
    }
  }

  override def originRDD(): RDD[_] = buffer

  override def cache(): Unit = {
    buffer.count()
    indexes.count()
    isCached = true
  }

  override def unpersist(): Unit = {
    buffer.unpersist()
    indexes.unpersist()
    isCached = false
  }
}


object RandomGenerator2 {

  var randomSeed = 1
  val generators = new ThreadLocal[RandomGenerator]()

  // scalastyle:off methodName
  def RNG: RandomGenerator = {
    if (generators.get() == null) {
      val rg = RandomGenerator.RNG.clone()
      rg.setSeed(randomSeed)
      generators.set(rg)
    }
    generators.get()
  }
  // scalastyle:on methodName

  def shuffle[T](data: Array[T]): Array[T] = {
    var i = 0
    val length = data.length
    while (i < length) {
      val exchange = RNG.uniform(0, length - i).toInt + i
      val tmp = data(exchange)
      data(exchange) = data(i)
      data(i) = tmp
      i += 1
    }
    data
  }
}

object TextClassifier {
  /**
   * The factory method to create a TextClassifier instance with WordEmbedding as
   * its first layer.
   *
   * @param classNum The number of text categories to be classified. Positive integer.
   * @param embeddingFile The path to the word embedding file.
   *                      Currently only the following GloVe files are supported:
   *                      "glove.6B.50d.txt", "glove.6B.100d.txt", "glove.6B.200d.txt",
   *                      "glove.6B.300d.txt", "glove.42B.300d.txt", "glove.840B.300d.txt".
   *                      You can download from: https://nlp.stanford.edu/projects/glove/.
   * @param wordIndex Map of word (String) and its corresponding index (integer).
   *                  The index is supposed to start from 1 with 0 reserved for unknown words.
   *                  During the prediction, if you have words that are not in the wordIndex
   *                  for the training, you can map them to index 0.
   *                  Default is null. In this case, all the words in the embeddingFile will
   *                  be taken into account and you can call
   *                  WordEmbedding.getWordIndex(embeddingFile) to retrieve the map.
   * @param sequenceLength The length of a sequence. Positive integer. Default is 500.
   * @param encoder The encoder for input sequences. String. "cnn" or "lstm" or "gru" are supported.
   *                Default is "cnn".
   * @param encoderOutputDim The output dimension for the encoder. Positive integer. Default is 256.
   * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
   */
  def apply[@specialized(Float, Double) T: ClassTag](
      classNum: Int,
      embeddingFile: String,
      wordIndex: Map[String, Int] = null,
      sequenceLength: Int = 500,
      encoder: String = "cnn",
      encoderOutputDim: Int = 256)(implicit ev: TensorNumeric[T]): TextClassifier[T] = {
    val embedding = WordEmbedding(embeddingFile, wordIndex, inputLength = sequenceLength)
    new TextClassifier[T](classNum, embedding.outputDim, sequenceLength, encoder,
      encoderOutputDim, embedding).build()
  }

  /**
   * The factory method to create a TextClassifier instance that takes word vectors as input.
   */
  @deprecated("Instead of using 'tokenLength', please pass the arguments 'embeddingFile' " +
    "and 'wordIndex' to construct a TextClassifier with WordEmbedding as the first layer.")
  def apply[@specialized(Float, Double) T: ClassTag](
      classNum: Int,
      tokenLength: Int,
      sequenceLength: Int,
      encoder: String,
      encoderOutputDim: Int)(implicit ev: TensorNumeric[T]): TextClassifier[T] = {
    new TextClassifier[T](classNum, tokenLength, sequenceLength, encoder, encoderOutputDim).build()
  }

  /**
   * This factory method is mainly for Python use.
   * Pass in a model to build the TextClassifier.
   * Note that if you use this factory method, arguments such as classNum, tokenLength, etc
   * should match the model definition to eliminate ambiguity.
   */
  private[zoo] def apply[@specialized(Float, Double) T: ClassTag](
      classNum: Int,
      embedding: Embedding[T],
      sequenceLength: Int,
      encoder: String,
      encoderOutputDim: Int,
      model: AbstractModule[Activity, Activity, T])
    (implicit ev: TensorNumeric[T]): TextClassifier[T] = {
    new TextClassifier[T](classNum, embedding.outputDim, sequenceLength,
      encoder, encoderOutputDim, embedding).addModel(model)
  }

  /**
   * Load an existing TextClassifier model (with weights).
   *
   * @param path The path for the pre-defined model.
   *             Local file system, HDFS and Amazon S3 are supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx".
   *             Amazon S3 path should be like "s3a://bucket/xxx".
   * @param weightPath The path for pre-trained weights if any. Default is null.
   * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
   */
  def loadModel[T: ClassTag](
      path: String,
      weightPath: String = null)(implicit ev: TensorNumeric[T]): TextClassifier[T] = {
    ZooModel.loadModel(path, weightPath).asInstanceOf[TextClassifier[T]]
  }
}
