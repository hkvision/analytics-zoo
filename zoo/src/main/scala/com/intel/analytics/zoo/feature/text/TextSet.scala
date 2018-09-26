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

package com.intel.analytics.zoo.feature.text

import java.io.File
import java.util

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset.{DataSet, Sample}
import com.intel.analytics.zoo.feature.common.Preprocessing
import com.intel.analytics.zoo.feature.text.TruncMode.TruncMode
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.log4j.Logger
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

/**
 * TextSet wraps a set of TextFeature.
 */
abstract class TextSet {
  /**
   * Transform from one TextSet to another.
   */
  def transform(transformer: Preprocessing[TextFeature, TextFeature]): TextSet

  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  def -> (transformer: Preprocessing[TextFeature, TextFeature]): TextSet = {
    this.transform(transformer)
  }

  /**
   * Whether it is a LocalTextSet.
   */
  def isLocal: Boolean

  /**
   * Whether it is a DistributedTextSet.
   */
  def isDistributed: Boolean

  /**
   * Convert to a LocalTextSet.
   */
  def toLocal(): LocalTextSet

  /**
   * Convert to a DistributedTextSet.
   */
  def toDistributed(sc: SparkContext = null): DistributedTextSet

  def toDistributed(sc: SparkContext, partitionNum: Int): DistributedTextSet

  /**
   * Convert TextSet to DataSet of Sample.
   */
  def toDataSet: DataSet[Sample[Float]]

  /**
   * Randomly split into array of TextSet with provided weights.
   * Only available for DistributedTextSet for now.
   */
  def randomSplit(weights: Array[Double]): Array[TextSet]

  /**
   * Tokenization on original text.
   * See Tokenizer for more details.
   */
  def tokenize(): TextSet = {
    transform(Tokenizer())
  }

  /**
   * Normalization on tokens.
   * See Normalizer for more details.
   */
  def normalize(): TextSet = {
    transform(Normalizer())
  }

  /**
   * Shape the sequence of tokens to a fixed length.
   * See SequenceShaper for more details.
   */
  def shapeSequence(
     len: Int,
     truncMode: TruncMode = TruncMode.pre,
     padElement: String = "##"): TextSet = {
    transform(SequenceShaper(len, truncMode, padElement))
  }

  /**
   * Map word tokens to index tokens.
   * All words occurred will be given an index and index starts from 1.
   * See WordIndexer for more details.
   *
   * After word2idx, you can get the wordIndex map by calling 'getWordIndex'.
   */
  def word2idx(): TextSet = {
    if (wordIndex != null) {
      TextSet.logger.info("wordIndex already exists. Using the existing wordIndex")
    } else {
      generateWordIndexMap()
    }
    transform(WordIndexer(wordIndex))
  }

  /**
   * Generate BigDL Sample.
   * See TextFeatureToSample for more details.
   */
  def genSample(): TextSet = {
    transform(TextFeatureToSample())
  }

  /**
   * Give each word occurred an index starting from 1.
   */
  protected def generateWordIndexMap(): Unit

  private var wordIndex: Map[String, Int] = _

  /**
   * Get the word index map of this TextSet.
   * If the TextSet hasn't been transformed from word to index, null will be returned.
   */
  def getWordIndex: Map[String, Int] = wordIndex

  def setWordIndex(map: Map[String, Int]): this.type = {
    wordIndex = map
    this
  }
}


object TextSet {

  val logger: Logger = Logger.getLogger(getClass)

  /**
   * Create a LocalTextSet from array of TextFeature.
   */
  def array(data: Array[TextFeature]): LocalTextSet = {
    new LocalTextSet(data)
  }

  /**
   * Create a DistributedTextSet from RDD of TextFeature.
   */
  def rdd(data: RDD[TextFeature]): DistributedTextSet = {
    new DistributedTextSet(data)
  }

  /**
   * Read text files as TextSet.
   * If sc is defined, read texts as DistributedTextSet from local file system or HDFS.
   * If sc is null, read texts as LocalTextSet from local file system.
   *
   * @param path String. Folder path to texts. The folder structure is expected to be the following:
   *             path
   *                ├── dir1 - text1, text2, ...
   *                ├── dir2 - text1, text2, ...
   *                └── dir3 - text1, text2, ...
   *             Under the target path, there ought to be N subdirectories (dir1 to dirN). Each
   *             subdirectory represents a category and contains all texts that belong to such
   *             category. Each category will be a given a label according to its position in the
   *             ascending order sorted among all subdirectories.
   *             All texts will be given a label according to the subdirectory where it is located.
   *             Labels start from 0.
   * @param sc An instance of SparkContext if any. Default is null.
   * @param minPartitions A suggestion value of the minimal partition number.
   *                      Integer. Default is 1. Only need to specify this when sc is not null.
   * @return TextSet.
   */
  def read(path: String, sc: SparkContext = null, minPartitions: Int = 1): TextSet = {
    val textSet = if (sc != null) {
      val fs = FileSystem.get(new Configuration())
      val categories = fs.listStatus(new Path(path)).map(_.getPath.getName).sorted
      logger.info(s"Found ${categories.length} classes.")
      // Labels of categories start from 0.
      val indices = categories.indices
      val categoryToLabel = categories.zip(indices).toMap
      val textRDD = sc.wholeTextFiles(path + "/*", minPartitions).map{case (p, text) =>
        val parts = p.split("/")
        val category = parts(parts.length - 2)
        TextFeature(text, label = categoryToLabel(category))
      }
      TextSet.rdd(textRDD)
    }
    else {
      val texts = ArrayBuffer[String]()
      val labels = ArrayBuffer[Int]()
      val categoryToLabel = new util.HashMap[String, Int]()
      val categoryPath = new File(path)
      require(categoryPath.exists(), s"$path doesn't exist. Please check your input path")
      val categoryPathList = categoryPath.listFiles().filter(_.isDirectory).toList.sorted
      categoryPathList.foreach { categoryPath =>
        val label = categoryToLabel.size()
        categoryToLabel.put(categoryPath.getName, label)
        val textFiles = categoryPath.listFiles()
          .filter(_.isFile).filter(_.getName.forall(Character.isDigit(_))).sorted
        textFiles.foreach { file =>
          val source = Source.fromFile(file, "ISO-8859-1")
          val text = try source.getLines().toList.mkString("\n") finally source.close()
          texts.append(text)
          labels.append(label)
        }
      }
      val textArr = texts.zip(labels).map{case (text, label) =>
        TextFeature(text, label)
      }.toArray
      TextSet.array(textArr)
    }
    textSet
  }

  // Given an array of words, zip each word with its corresponding index.
  // Return a wordIndex map. Index starts from 1.
  def zipWordIndex(words: Array[String]): Map[String, Int] = {
    val indexes = Range(1, words.length + 1)
    words.zip(indexes).toMap
  }
}


class LocalTextSet(var array: Array[TextFeature]) extends TextSet {

  override def transform(transformer: Preprocessing[TextFeature, TextFeature]): TextSet = {
    array = transformer.apply(array.toIterator).toArray
    this
  }

  override def isLocal: Boolean = true

  override def isDistributed: Boolean = false

  override def toLocal(): LocalTextSet = {
    this
  }

  override def toDistributed(sc: SparkContext): DistributedTextSet = {
    new DistributedTextSet(sc.parallelize(array))
  }

  override def toDistributed(sc: SparkContext, partitionNum: Int): DistributedTextSet = {
    new DistributedTextSet(sc.parallelize(array, partitionNum))
  }

  override def toDataSet: DataSet[Sample[Float]] = {
    DataSet.array(array.map(_[Sample[Float]](TextFeature.sample)))
  }

  override def randomSplit(weights: Array[Double]): Array[TextSet] = {
    throw new UnsupportedOperationException("LocalTextSet doesn't support randomSplit for now")
  }

  override def generateWordIndexMap(): Unit = {
    val frequencies = array.flatMap(_[Array[String]](TextFeature.tokens)).distinct
    val wordIndex = TextSet.zipWordIndex(frequencies)
    setWordIndex(wordIndex)
  }
}


class DistributedTextSet(var rdd: RDD[TextFeature]) extends TextSet {

  override def transform(transformer: Preprocessing[TextFeature, TextFeature]): TextSet = {
    rdd = transformer(rdd)
    this
  }

  override def isLocal: Boolean = false

  override def isDistributed: Boolean = true

  override def toLocal(): LocalTextSet = {
    new LocalTextSet(rdd.collect())
  }

  override def toDistributed(sc: SparkContext = null): DistributedTextSet = {
    this
  }

  override def toDistributed(sc: SparkContext, partitionNum: Int): DistributedTextSet = {
    this
  }

  override def toDataSet: DataSet[Sample[Float]] = {
    DataSet.rdd(rdd.map(_[Sample[Float]](TextFeature.sample)))
  }

  override def randomSplit(weights: Array[Double]): Array[TextSet] = {
    rdd.randomSplit(weights).map(TextSet.rdd)
  }

  override def generateWordIndexMap(): Unit = {
    val words = rdd.flatMap(_[Array[String]](TextFeature.tokens)).distinct().collect()
    rdd.persist()
    val wordIndex = TextSet.zipWordIndex(words)
    val wordIndexBC = rdd.sparkContext.broadcast(wordIndex)
    setWordIndex(wordIndexBC.value)
  }
}
