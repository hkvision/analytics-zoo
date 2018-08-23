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

import com.johnsnowlabs.nlp.annotator.NormalizerModel
import org.apache.spark.ml.Transformer

/**
 * Removes all dirty characters from tokens following a regex pattern.
 * Input key: TextFeature.tokens
 * Output key: Can be specified by outKey. Default is TextFeature.tokens.
 *             By default original tokens will be replaced by normalized tokens.
 */
class Normalizer(val outKey: String = TextFeature.tokens) extends SparkNLPTransformer {

  override def labor: Transformer = {
    new NormalizerModel().setLowercase(true)
      .setInputCols(Array("tokens"))
      .setOutputCol("normalized")
      .setPatterns(Array("[^\\pL+]"))
      .setSlangDict(Map.empty[String, String])
  }
}

object Normalizer {
  def apply(): Normalizer = {
    new Normalizer()
  }
}
