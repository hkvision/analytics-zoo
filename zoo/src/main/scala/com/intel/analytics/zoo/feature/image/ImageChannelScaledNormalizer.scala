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

package com.intel.analytics.zoo.feature.image

import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.transform.vision.image.augmentation.ChannelScaledNormalizer
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat

/**
 * Channel normalization with scale factor.
 *
 * @param meanR Integer. The mean value for channel R.
 * @param meanG Integer. The mean value for channel G.
 * @param meanB Integer. The mean value for channel R.
 * @param scale Double. The scale value applied for all channels.
 */
class ImageChannelScaledNormalizer(meanR: Int, meanG: Int,
                                   meanB: Int, scale: Double) extends ImageProcessing {
  private val internalTransformer = new InternalChannelScaledNormalizer(meanR,
    meanG, meanB, scale)

  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    internalTransformer.apply(prev)
  }

  override def transformMat(feature: ImageFeature): Unit = {
    internalTransformer.transformMat(feature)
  }
}

object ImageChannelScaledNormalizer {
  def apply(meanR: Int, meanG: Int,
            meanB: Int, scale: Double): ImageChannelScaledNormalizer = {
    new ImageChannelScaledNormalizer(meanR, meanG, meanB, scale)
  }
}

// transformMat in BigDL RandomCropper is protected and can't be directly accessed.
// Thus add an InternalChannelScaledNormalizer here to override transformMat and make it accessible.
private[image] class InternalChannelScaledNormalizer(meanR: Int, meanG: Int,
                                                     meanB: Int, scale: Double)
  extends ChannelScaledNormalizer(meanR, meanG, meanB, scale) {

  override def transformMat(feature: ImageFeature): Unit = {
//    val mat = feature.opencvMat()
//    val toFloats = OpenCVMat.toFloatPixels(mat)
//    val content = toFloats._1
//    if (content.length % 3 != 0) {
//      print(mat.channels())
//      println(feature.uri())
//    }
//    super.transformMat(feature)
    val mat = feature.opencvMat()
    val floats = new Array[Float](mat.height() * mat.width() * 3)
    val toFloats = OpenCVMat.toFloatPixels(mat, floats)
    val content = toFloats._1
    require(content.length % 3 == 0, "Content should be multiple of 3 channels")
    var i = 0
    val frameLength = content.length / 3
    val height = toFloats._2
    val width = toFloats._3
    val bufferContent = new Array[Float](width * height * 3)

    val channels = 3
    val mean = Array(meanR, meanG, meanB)
    var c = 0
    while (c < channels) {
      i = 0
      while (i < frameLength) {
        val data_index = c * frameLength + i
        bufferContent(data_index) = ((content(data_index) - mean(c)) * scale).toFloat
        i += 1
      }
      c += 1
    }
    if (mat != null) {
      mat.release()
    }
    val newMat = OpenCVMat.fromFloats(bufferContent, height, width)
    feature(ImageFeature.mat) = newMat
  }
}
