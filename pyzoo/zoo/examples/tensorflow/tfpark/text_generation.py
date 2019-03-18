from __future__ import absolute_import, division, print_function

import tensorflow as tf
# tf.enable_eager_execution()

import numpy as np
import os
import time

from zoo import init_nncontext
from zoo.tfpark import KerasModel, variable_creator_scope

sc = init_nncontext("Text Generation Example")

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print ('Length of text: {} characters'.format(len(text)))

# Take a look at the first 250 characters in text
print(text[:250])

# The unique characters in the file
vocab = sorted(set(text))
print('{} unique characters'.format(len(vocab)))

# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')

# Show how the first 13 characters from the text are mapped to integers
print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epoch = len(text)//seq_length

text_as_int = text_as_int[:len(text)//(seq_length+1)*(seq_length+1)]  # drop remainder
sequences = np.reshape(np.asarray(list(text_as_int)), [-1, seq_length+1])


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = list(map(split_input_target, sequences))
input_example, target_example = zip(*dataset)
input_example = np.asarray(input_example)
target_example = np.asarray(target_example)

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

BATCH_SIZE = 64
steps_per_epoch = examples_per_epoch//BATCH_SIZE

with variable_creator_scope():
    # words_input = tf.keras.layers.Input(shape=(None,), name='words_input')
    # embedding_layer = tf.keras.layers.Embedding(vocab_size,
    #                             embedding_dim, name='word_embedding')
    # word_embeddings = embedding_layer(words_input)
    # lstm = tf.keras.layers.LSTM(rnn_units, return_sequences=True)(word_embeddings)
    # output = tf.keras.layers.Dense(vocab_size)(lstm)
    # model = tf.keras.Model(inputs=words_input, outputs=output)
    model = tf.keras.Sequential([
      tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                batch_input_shape=[BATCH_SIZE, None]),
      tf.keras.layers.LSTM(rnn_units,
                          return_sequences=True),
      tf.keras.layers.Dense(vocab_size)
    ])

model.summary()


def loss(labels, logits):
    return tf.keras.backend.sparse_categorical_crossentropy(labels, logits, from_logits=True)


model.compile(optimizer=tf.train.AdamOptimizer(), loss=loss)

EPOCHS=3

keras_model = KerasModel(model)
keras_model.fit(input_example, target_example,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              # steps_per_epoch=steps_per_epoch,
              distributed=True)

