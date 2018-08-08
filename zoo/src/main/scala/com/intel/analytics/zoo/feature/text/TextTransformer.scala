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

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.zoo.feature.common.Preprocessing

abstract class TextTransformer extends Transformer[TextFeature, TextFeature]
  with Preprocessing[TextFeature, TextFeature]{

  def transform(feature: TextFeature): TextFeature

  override def apply(prev: Iterator[TextFeature]): Iterator[TextFeature] = {
    prev.map(transform)
  }
}
