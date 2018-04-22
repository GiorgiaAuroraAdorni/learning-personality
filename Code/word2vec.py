#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import basic_w2v as w2v
import TFRecord_dataset as exp_TFR

test_filename = "Dataset/TFRecords/test.tfrecords.gz"
dataset = tf.data.TFRecordDataset(test_filename, compression_type = 'GZIP')
# # map takes a python function and applies it to every sample
corpus = dataset.map(exp_TFR.decode)
#corpus = corpus.padded_batch(10000, padded_shapes = ([None], [None]))

#corpus = dataset.apply(tf.contrib.data.scan(tf.constant([], dtype = tf.string), w2v.get_corpus))
#corpus = dataset.apply(tf.contrib.data.scan(([], dtype = tf.string), w2v.get_corpus))

corpus_iterator = corpus.make_one_shot_iterator()
next = corpus_iterator.get_next()

# ## Creo un iteratore a partire dal dataset
# iterator = dataset.make_one_shot_iterator()
# ## Tensore simile a un placeholder
# next_element = iterator.get_next()

##################

# # 2: Build the dictionary and replace rare words with UNK token.

# variables: (?)
# - dictionary - map of words(strings) to their codes(integers)
# - reverse_dictionary - maps codes(integers) to words(strings)

# Extract the top 60000 most common words to include in our embedding vector
vocab_size = 60000

# Gather together all the unique words and index them with a unique integer value – 
# We’ll use a dictionary to do this

# Loop through every word in the dataset (vocabulary variable) and assign it to 
# the unique integer word identified.  

# Because we are restricting our vocabulary to only 60000 words, 
# any words not within the top 60000 most common words will be marked with "-1" (unknown). 

# Load the dictionary populated by keys corresponding to each unique word
table = tf.contrib.lookup.index_table_from_file(
    vocabulary_file = "Dataset/Vocabulary/compact_dict.txt", vocab_size = vocab_size, 
    key_column_index = 1, delimiter = ' ')

# Create a reverse_table that allows us to look up a word based on its unique integer identifier, 
# rather than looking up the identifier based on the word.  
reverse_table = tf.contrib.lookup.index_to_string_table_from_file(
    vocabulary_file = "Dataset/Vocabulary/compact_dict.txt", vocab_size = vocab_size, 
    value_column_index = 1, delimiter = ' ')


# A list called data is created, which will be the same length as words but 
# instead of being a list of individual words, it will instead be a list of integers 
# – with each word now being represented by the unique integer that was assigned to this word in dictionary.  
id, text = next
data = table.lookup(text)
rev_text = reverse_table.lookup(data)

data_big = []
window_size = 5

# x_train = [] 
# y_train = [] 

indices = bho
depth = vocab_size

with tf.Session() as sess:
    tf.tables_initializer().run()
    #print(data.eval())
    
    while True:
        try:
            text_, ids_, rev_text_ = sess.run([text, data, rev_text])

            # for word_index, word in enumerate(ids_):
            #     for nb_word in ids_[max(word_index - window_size, 0) : min(word_index + window_size, len(ids_)) + 1]:
            #
            #             if nb_word != word:
            #                 data_big.append([word, nb_word])
            #                 x_train.append(w2v.to_one_hot(word, vocab_size))
            #                 y_train.append(w2v.to_one_hot(nb_word, vocab_size))
            #                 #print(x_train)
            #                 print([word, nb_word])

            #id = tf.constant([42], dtype=tf.int64)
            #word = reverse_table.lookup(id)
            #print(sess.run(word))

            print('\n\n')

            #print(text_, ids_, rev_text_)
        except tf.errors.OutOfRangeError:
            break

# x_train = np.asarray(x_train) 

# 3: Function to generate a training batch for the skip-gram model.
data_index = 0

batch_size = 128
#embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2  

#print(w2v.generate_batch(data, batch_size, num_skips, skip_window))


# window size: 5


batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0],
        reverse_dictionary[labels[i, 0]])







# 4: Build and train a skip-gram model.
# 5: Begin training.

# Our method integrates GloVe features with Gaussian Process regression as the learning algorithm.

# 6: Visualize the embeddin