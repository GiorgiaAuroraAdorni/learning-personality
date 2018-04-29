#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from load_ocean import get_ocean_value

# Build the dictionary
# Extract the top 60000 most common words to include in our embedding vector
vocab_file_path = "Dataset/Vocabulary/resized_dictionary.txt"
vocab_size = 60000

# Gather together all the unique words and index them with a unique integer value
# Loop through every word in the dataset and assign it to the unique integer word identified.
# Any words not within the top 60000 most common words will be marked with "-1" and replace with "UNK" token

# Load the dictionary populated by keys corresponding to each unique word
table = tf.contrib.lookup.index_table_from_file(vocabulary_file=vocab_file_path,
                                                    vocab_size=vocab_size, key_column_index=1,
                                                    delimiter=' ')

# Create a reverse_table that allows us to look up a word based on its unique integer identifier,
# rather than looking up the identifier based on the word.
# reverse_table = tf.contrib.lookup.index_to_string_table_from_file(vocabulary_file=vocab_file_path,
#                                                                   vocab_size=vocab_size, value_column_index=1,
#                                                                   delimiter=' ')

# Load ocean dictionary
ocean_dict_file_path = "Dataset/Vocabulary/ocean_dict.txt"
ocean_dict_size = 636

# Ocean lookup-table
ocean_table = tf.contrib.lookup.index_table_from_file(vocabulary_file=ocean_dict_file_path,
                                                      vocab_size=ocean_dict_size, key_column_index=0,
                                                      delimiter='\t')

file = tf.read_file(ocean_dict_file_path)
lines = tf.string_split([file], '\n').values

# Create a variable.
value = tf.Variable(get_ocean_value(lines), validate_shape=False)


def extract_features(ids, text):
    """ Function that extract label and feature, for the model, from dataset. """

    data = table.lookup(text)                 # integers lookup-table
    # rev_text = reverse_table.lookup(data)   # words lookup-table

    # Create the bag-of-words vector
    data = tf.contrib.framework.sort(data)
    data, idx = tf.unique(data + 1, out_idx=tf.int64)

    bow_vector = tf.sparse_to_dense(data,              # sparse_indices
                                    [vocab_size + 1],  # output_shape
                                    1)                 # sparse_values
    # Remove the first element
    bow_vector = bow_vector[1:]

    features = {'bow_vector': bow_vector}

    ocean = ocean_table.lookup(text)

    # Create ocean vector
    ocean = ocean + 1
    ocean_new_table = tf.nn.embedding_lookup(value, ocean)

    # Compute the mean of the non_zero ocean vector
    non_zero = tf.count_nonzero(ocean, 0, dtype=tf.float32)
    ocean_vector = tf.reduce_sum(ocean_new_table, 0) / non_zero

    return features, ocean_vector
