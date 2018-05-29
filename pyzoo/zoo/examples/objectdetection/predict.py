#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse

from zoo.common.nncontext import get_nncontext
from zoo.models.image.objectdetection import *

sc = get_nncontext(create_spark_conf().setAppName("Object Detection Example"))

parser = argparse.ArgumentParser()
parser.add_argument('model_path', help="Path where the model is stored")
parser.add_argument('img_path', help="Path where the images are stored")
parser.add_argument('partition_num', type=int, default=4, help="Path to store the detection results")


def predict(model_path, img_path, partition_num=4):
    model = ObjectDetector.load_model(model_path)
    image_set = ImageSet.read(img_path, sc, partition_num)
    output = model.predict_image_set(image_set)
    result = output.get_predict().collect()
    print(result[0][1])


if __name__ == "__main__":
    args = parser.parse_args()
    predict(args.model_path, args.img_path, args.partition_num)
