#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import numpy as np
import os
import time

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
#print(path_to_file)

# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
#print ('Length of text: {} characters'.format(len(text)))

# The unique characters in the file
vocab = sorted(set(text))
#print ('{} unique characters'.format(len(vocab)))



##Vectorize the text
# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
#print("indexed characters: \n {}".format(char2idx))
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

"""print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')"""

# Show how the first 13 characters from the text are mapped to integers
#print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))



##Create training examples and targets
# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

"""for i in char_dataset.take(5):
  print(idx2char[i.numpy()])"""


#The batch method lets us easily convert these individual characters to sequences of the desired size.
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(5):
  print(repr(''.join(idx2char[item.numpy()])))
  
