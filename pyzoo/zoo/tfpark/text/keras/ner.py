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

import tensorflow as tf
from tensorflow.keras.layers import Lambda, Bidirectional, LSTM, Input, Dense
from tensorflow.keras.models import Model
import nlp_architect.models.ner_crf as ner_model
from bert import modeling
from zoo.tfpark.text.keras.text_model import TextKerasModel
from zoo.tfpark import KerasModel


class NER(TextKerasModel):
    """
    The model used for named entity recognition using Bidirectional LSTM with
    Conditional Random Field (CRF) sequence classifier.

    This model has two inputs:
    - word indices of shape (batch, sequence_length)
    - character indices of shape (batch, sequence_length, word_length)
    This model outputs entity tags of shape (batch, sequence_length, num_entities).

    :param num_entities: Positive int. The number of entity labels to be classified.
    :param word_vocab_size: Positive int. The size of the word dictionary.
    :param char_vocab_size: Positive int. The size of the character dictionary.
    :param word_length: Positive int. The max word length in characters. Default is 12.
    :param word_emb_dim: Positive int. The dimension of word embeddings. Default is 100.
    :param char_emb_dim: Positive int. The dimension of character embeddings. Default is 30.
    :param tagger_lstm_dim: Positive int. The hidden size of tagger Bi-LSTM layers. Default is 100.
    :param dropout: Dropout rate. Default is 0.5.
    :param crf_mode: String. CRF operation mode. Either 'reg' or 'pad'. Default is 'reg'.
                     'reg' for regular full sequence learning (all sequences have equal length).
                     'pad' for supplied sequence lengths (useful for padded sequences).
                     For 'pad' mode, a third input for sequence_length (batch, 1) is needed.
    :param optimizer: Optimizer to train the model. If not specified, it will by default
                      to be tf.keras.optimizers.Adam(0.001, clipnorm=5.).
    """
    def __init__(self, num_entities, word_vocab_size, char_vocab_size, word_length=12,
                 word_emb_dim=100, char_emb_dim=30, tagger_lstm_dim=100, dropout=0.5,
                 crf_mode='reg', optimizer=None):
        super(NER, self).__init__(ner_model.NERCRF(use_cudnn=False), optimizer,
                                  word_length=word_length,
                                  target_label_dims=num_entities,
                                  word_vocab_size=word_vocab_size,
                                  char_vocab_size=char_vocab_size,
                                  word_embedding_dims=word_emb_dim,
                                  char_embedding_dims=char_emb_dim,
                                  tagger_lstm_dims=tagger_lstm_dim,
                                  dropout=dropout,
                                  crf_mode=crf_mode)
        # Remark: In nlp-architect NERCRF.build(..), word_lstm_dims is never used.
        # Thus, removed this argument here to avoid ambiguity.

    @staticmethod
    def load_model(path):
        """
        Load an existing NER model (with weights) from HDF5 file.

        :param path: String. The path to the pre-defined model.
        :return: NER.
        """
        labor = ner_model.NERCRF(use_cudnn=False)
        model = TextKerasModel._load_model(labor, path)
        model.__class__ = NER
        return model


class BERTNER(KerasModel):
    def __init__(self, num_entities, bert_config_file, init_checkpoint=None,
                 use_one_hot_embeddings=False):
        self.bert_config_file = bert_config_file
        self.init_checkpoint = init_checkpoint
        self.use_one_hot_embeddings = use_one_hot_embeddings

        def bert(inputs):
            bert_config = modeling.BertConfig.from_json_file(self.bert_config_file)
            model = modeling.BertModel(
                config=bert_config,
                is_training=True,
                input_ids=inputs[0],
                input_mask=inputs[1],
                token_type_ids=inputs[2],
                use_one_hot_embeddings=self.use_one_hot_embeddings)
            output_layer = model.get_all_encoder_layers()[-4:]
            return tf.concat(output_layer, axis=-1)
        input_id = Input(shape=(128, ), name="input_id", dtype='int32')
        input_mask = Input(shape=(128, ), name="input_mask", dtype='int32')
        token_type_ids = Input(shape=(128, ), name="token_type_ids", dtype='int32')
        embeddings = Lambda(bert, output_shape=(128, 4*768), trainable=False)([input_id, input_mask, token_type_ids])
        # bilstm1 = Bidirectional(LSTM(32, return_sequences=True))(embeddings)
        # bilstm2 = Bidirectional(LSTM(32, return_sequences=True))(bilstm1)
        # output = Dense(num_entities, activation="softmax")(bilstm2)
        model = Model([input_id, input_mask, token_type_ids], embeddings)
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(0.001, clipnorm=5.))
        self.model = model
        super(BERTNER, self).__init__(model)
