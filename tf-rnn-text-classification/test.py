#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

print(tf.version.VERSION)
import matplotlib.pyplot as plt

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string], '')
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

def pad_to_size(vec, size):
  zeros = [0] * (size - len(vec))
  vec.extend(zeros)
  return vec

def sample_predict(sentence, pad):
  encoded_sample_pred_text = encoder.encode(sample_pred_text)

  if pad:
    encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
  encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
  predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))

  return (predictions)

#https://www.tensorflow.org/tutorials/keras/save_and_load
model = tf.keras.models.load_model('./my_model.h5')

dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']
encoder = info.features['text'].encoder

# predict on a sample text without padding.
#sample_pred_text = ('The movie was cool. The animation and the graphics '
#                    'were out of this world. I would recommend this movie.')

print("Yorum yazın: \n")
sample_pred_text = input()
predictions = sample_predict(sample_pred_text, pad=False)

print (predictions)




# Check its architecture
#new_model.summary()
