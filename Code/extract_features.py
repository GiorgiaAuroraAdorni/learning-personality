#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from load_ocean import get_ocean_value


def extract_features(ids, text, table, ocean_table):
    """ Function that extract labels and features, for the model, from dataset. """

    ocean_dict_file_path = "Dataset/Vocabulary/ocean_dict.txt"
    vocab_size = 60000

    file = tf.read_file(ocean_dict_file_path)
    lines = tf.string_split([file], '\n').values

    # Create a variable.
    # value = tf.Variable(get_ocean_value(lines), validate_shape=False, name='value_variable')
    value = get_ocean_value(lines)

    data = table.lookup(text)                 # integers lookup-table
    # rev_text = reverse_table.lookup(data)   # words lookup-table

    # Create the bag-of-words vector
    data = tf.contrib.framework.sort(data)
    data, idx = tf.unique(data + 1, out_idx=tf.int64)

    bow_vector = tf.sparse_to_dense(data,              # sparse_indices
                                    [vocab_size + 1],  # output_shape
                                    1.0)               # sparse_values

    # Remove the first element
    bow_vector = bow_vector[1:]

    features = {'bow_vector': bow_vector}
    # features = bow_vector

    ocean = ocean_table.lookup(text)

    # Create ocean vector
    ocean = ocean + 1
    ocean_new_table = tf.nn.embedding_lookup(value, ocean)

    # Compute the mean of the non_zero ocean vector
    non_zero = tf.count_nonzero(ocean, 0, dtype=tf.float32)
    ocean_vector = tf.reduce_sum(ocean_new_table, 0) / non_zero

    #ocean_value = tf.reduce_mean(ocean_vector)

    return features, ocean_vector  #, ocean_value 
