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

import sys
import six
from bigdl.util.common import JavaValue, callBigDlFunc

if sys.version >= '3':
    long = int
    unicode = str


class TextFeature(JavaValue):
    """
    Each TextFeature keeps information of a single text record.
    It can include various status of a text,
    e.g. original text content, category label, tokens, index representation
    of tokens, BigDL Sample representation, prediction result and so on.
    """
    def __init__(self, text=None, label=None, jvalue=None, bigdl_type="float"):
        self.bigdl_type = bigdl_type
        if jvalue:
            self.value = jvalue
        else:
            assert isinstance(text, six.string_types), "text of a TextFeature should be a string"
            if label is not None:
                self.value = callBigDlFunc(bigdl_type, JavaValue.jvm_class_constructor(self),
                                           text, int(label))
            else:
                self.value = callBigDlFunc(bigdl_type, JavaValue.jvm_class_constructor(self),
                                           text)

    def get_text(self):
        """
        Get the text content of the TextFeature.

        :return: String
        """
        return callBigDlFunc(self.bigdl_type, "textFeatureGetText", self.value)

    def get_label(self):
        """
        Get the label of the TextFeature.
        If no label is stored, -1 will be returned.

        :return: Int
        """
        return callBigDlFunc(self.bigdl_type, "textFeatureGetLabel", self.value)

    def has_label(self):
        """
        Whether the TextFeature contains label.

        :return: Boolean
        """
        return callBigDlFunc(self.bigdl_type, "textFeatureHasLabel", self.value)

    def set_label(self, label):
        """
        Set the label for the TextFeature.

        :param label: Int
        :return: The TextFeature with label.
        """
        self.value = callBigDlFunc(self.bigdl_type, "textFeatureSetLabel", self.value, int(label))
        return self

    def get_tokens(self):
        """
        Get the tokens of the TextFeature.
        If text hasn't been segmented, None will be returned.

        :return: List of String
        """
        return callBigDlFunc(self.bigdl_type, "textFeatureGetTokens", self.value)

    def get_sample(self):
        """
        Get the Sample representation of the TextFeature.
        If the TextFeature hasn't been transformed to Sample, None will be returned.

        :return: BigDL Sample
        """
        return callBigDlFunc(self.bigdl_type, "textFeatureGetSample", self.value)

    def keys(self):
        """
        Get the keys that the TextFeature contains.

        :return: List of String
        """
        return callBigDlFunc(self.bigdl_type, "textFeatureGetKeys", self.value)
