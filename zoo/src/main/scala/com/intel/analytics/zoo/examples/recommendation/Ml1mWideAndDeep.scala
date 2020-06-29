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

package com.intel.analytics.zoo.examples.recommendation

import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{Adam, PrecisionRecallAUC, Top1Accuracy}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.models.recommendation._
import com.intel.analytics.zoo.pipeline.api.keras.metrics.AUC
import com.intel.analytics.zoo.pipeline.api.keras.objectives.{BinaryCrossEntropy, SparseCategoricalCrossEntropy}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.rdd.RDD

case class User(userId: Int, gender: String, age: Int, occupation: Int)

case class Item(itemId: Int, title: String, genres: String)

object Ml1mWideAndDeep {

  def run(params: WNDParams): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().setAppName("WideAndDeepExample")
    val sc = NNContext.initNNContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)

    val (ratingsDF, userDF, itemDF, userCount, itemCount) =
      loadPublicData(sqlContext, params.inputDir)

    ratingsDF.groupBy("label").count().show()
    val bucketSize = 100
    val localColumnInfo = ColumnFeatureInfo(
      wideBaseCols = Array("occupation", "gender"),
      wideBaseDims = Array(21, 3))

    val wideAndDeep: WideAndDeep[Float] = WideAndDeep[Float](
      params.modelType,
      numClasses = 1,
      columnInfo = localColumnInfo)

    val isImplicit = false
    val featureRdds =
      assemblyFeature(isImplicit, ratingsDF, userDF, itemDF, localColumnInfo, params.modelType)

    val Array(trainpairFeatureRdds, validationpairFeatureRdds) =
      featureRdds.randomSplit(Array(0.8, 0.2))
    val trainRdds = trainpairFeatureRdds.map(x => x.sample)
    val validationRdds = validationpairFeatureRdds.map(x => x.sample)

    val optimMethod = new Adam[Float](
      learningRate = 1e-2,
      learningRateDecay = 1e-5)

    wideAndDeep.compile(optimizer = optimMethod,
      loss = BinaryCrossEntropy[Float](),
      metrics = List(new PrecisionRecallAUC[Float]())
    )
    wideAndDeep.fit(trainRdds, batchSize = params.batchSize,
      nbEpoch = params.maxEpoch, validationData = validationRdds)

    val results = wideAndDeep.predict(validationRdds)
    results.take(5).foreach(println)

    val resultsClass = wideAndDeep.predictClass(validationRdds)
    resultsClass.take(5).foreach(println)

    val userItemPairPrediction = wideAndDeep.predictUserItemPair(validationpairFeatureRdds)
    userItemPairPrediction.take(50).foreach(println)

    val userRecs = wideAndDeep.recommendForUser(validationpairFeatureRdds, 3)
    val itemRecs = wideAndDeep.recommendForItem(validationpairFeatureRdds, 3)

    userRecs.take(10).foreach(println)
    itemRecs.take(10).foreach(println)

    println("finished...")
    sc.stop()
  }

  def loadPublicData(sqlContext: SQLContext, dataPath: String):
  (DataFrame, DataFrame, DataFrame, Int, Int) = {
    import sqlContext.implicits._
    val ratings = sqlContext.read.text(dataPath + "/ratings.dat").as[String]
      .map(x => {
        val line = x.split("::").map(n => n.toInt)
        Rating(line(0), line(1), line(2))
      }).toDF()
    val userDF = sqlContext.read.text(dataPath + "/users.dat").as[String]
      .map(x => {
        val line = x.split("::")
        User(line(0).toInt, line(1), line(2).toInt, line(3).toInt)
      }).toDF()
    val itemDF = sqlContext.read.text(dataPath + "/movies.dat").as[String]
      .map(x => {
        val line = x.split("::")
        Item(line(0).toInt, line(1), line(2).split('|')(0))
      }).toDF()

    val minMaxRow = ratings.agg(max("userId"), max("itemId")).collect()(0)
    val (userCount, itemCount) = (minMaxRow.getInt(0), minMaxRow.getInt(1))

    (ratings, userDF, itemDF, userCount, itemCount)
  }

  // convert features to RDD[Sample[Float]]
  def assemblyFeature(isImplicit: Boolean = false,
                      ratingDF: DataFrame,
                      userDF: DataFrame,
                      itemDF: DataFrame,
                      columnInfo: ColumnFeatureInfo,
                      modelType: String): RDD[UserItemFeature[Float]] = {

    // age and gender as cross features, gender its self as wide base features
    val genderUDF = udf(Utils.categoricalFromVocabList(Array("F", "M")))
    val bucketUDF = udf(Utils.buckBucket(100))
    val genresList = Array("Crime", "Romance", "Thriller", "Adventure", "Drama", "Children's",
      "War", "Documentary", "Fantasy", "Mystery", "Musical", "Animation", "Film-Noir", "Horror",
      "Western", "Comedy", "Action", "Sci-Fi")
    val genresUDF = udf(Utils.categoricalFromVocabList(genresList))

    val userDfUse = userDF
      .withColumn("age-gender", bucketUDF(col("age"), col("gender")))
      .withColumn("gender", genderUDF(col("gender")))

    // genres as indicator
    val itemDfUse = itemDF
      .withColumn("genres", genresUDF(col("genres")))

    val unioned = if (isImplicit) {
      val negativeDF = Utils.getNegativeSamples(ratingDF)
      negativeDF.unionAll(ratingDF.withColumn("label", lit(2)))
    }
    else ratingDF

    // userId, itemId as embedding features
    val joined = unioned
      .join(itemDfUse, Array("itemId"))
      .join(userDfUse, Array("userId"))
      .select(unioned("userId"), unioned("itemId"), col("label"), col("gender"), col("age"),
        col("occupation"), col("genres"), col("age-gender"))

    // 3|261197|
    // 5|226310|
    val filtered = joined.filter(joined("label") === 3 || joined("label") === 5)

    val relabel = (s: Int) => {
      if (s == 3) 0 else 1
    }
    val relabelUDF = udf(relabel)
    val relabeled = filtered.withColumn("label", relabelUDF(col("label")))

    val rddOfSample = relabeled.rdd.map(r => {
      val uid = r.getAs[Int]("userId")
      val iid = r.getAs[Int]("itemId")
      UserItemFeature(uid, iid, Utils.row2Sample(r, columnInfo, modelType))
    })
    rddOfSample
  }
}
