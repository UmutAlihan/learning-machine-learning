#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import numpy as np
import os
import time

#path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
#path_to_file = "/home/uad/l-m-l/tf-rnn-text-generation/dervisesor/cemalnur-kitaplar/merged.txt"
path_to_file = "/home/uad/datasets/cemalnur-kitaplar/preprocessed.txt"
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

#for item in sequences.take(5):
#  print(repr(''.join(idx2char[item.numpy()])))


#For each sequence, duplicate and shift it to form the input and target text by using the map method to apply a simple function to each batch:
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

#for input_example, target_example in  dataset.take(1):
#  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
#  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))


#Each index of these vectors are processed as one time step. For the input at time step 0, the model receives the index for "F" and trys to predict the index for "i" as the next character. At the next timestep, it does the same thing but the RNN considers the previous step context in addition to the current input character
#for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
#    print("Step {:4d}".format(i))
#    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
#    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))


##Create training batches
#before feeding this data into the model, we need to shuffle the data and pack it into batches
# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
#dataset


##Build The Model
#For this simple example three layers are used to define our model:
#tf.keras.layers.Embedding: The input layer. A trainable lookup table that will map the numbers of each character to a vector with embedding_dim dimensions
#tf.keras.layers.GRU: A type of RNN with size units=rnn_units (You can also use a LSTM layer here.)
#tf.keras.layers.Dense: The output layer, with vocab_size outputs
# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size, activation='relu')
  ])
  return model

#model = build_model(
#  vocab_size = len(vocab),
#  embedding_dim=embedding_dim,
#  rnn_units=rnn_units,
#  batch_size=BATCH_SIZE)


##Try the model
#for input_example_batch, target_example_batch in dataset.take(1):
#  example_batch_predictions = model(input_example_batch)
#  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

#In the above example the sequence length of the input is 100 but the model can be run on inputs of any length:
#model.summary()

#To get actual predictions from the model we need to sample from the output distribution, to get actual character indices. This distribution is defined by the logits over the character vocabulary.
#Try it for the first example in the batch:
#sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
#This gives us, at each timestep, a prediction of the next character index
#sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

#Decode these to see the text predicted by this untrained model:
#print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
#print()
#print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))


##Train the model
##Attach an optimizer, and a loss function
#def loss(labels, logits):
#  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

#example_batch_loss  = loss(target_example_batch, example_batch_predictions)
#print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
#print("scalar_loss:      ", example_batch_loss.numpy().mean())

#Configure the training procedure using the tf.keras.Model.compile method. We'll use tf.keras.optimizers.Adam with default arguments and the loss function.
#model.compile(optimizer='adam', loss=loss)

##Configure checkpoints
# Directory where the checkpoints will be saved
checkpoint_dir = './3-tc'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

##Execute the training
#EPOCHS=50
#history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])


##Generate text
#Restore the latest checkpoint
#keep this prediction step simple, use a batch size of 1.
#Because of the way the RNN state is passed from timestep to timestep, the model only accepts a fixed batch size once built
tf.train.latest_checkpoint(checkpoint_dir)

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))


##The prediction loop
#The following code block generates the text:
#It Starts by choosing a start string, initializing the RNN state and setting the number of characters to generate.
#Get the prediction distribution of the next character using the start string and the RNN state.
#Then, use a categorical distribution to calculate the index of the predicted character. Use this predicted character as our next input to the model.
#The RNN state returned by the model is fed back into the model so that it now has more context, instead than only one word. After predicting the next word, the modified RNN states are again fed back into the model, which is how it learns as it gets more context from the previously predicted words.
def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 144

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.3

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the word returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))


print(generate_text(model, start_string=u"Ben bir sey yaziyorum"))

