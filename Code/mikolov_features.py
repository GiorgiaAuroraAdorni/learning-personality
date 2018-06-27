#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from load_ocean import get_ocean_value


def extract_mikolov_sentences(text, table):
    """ Function that extract labels and features, for the model, from dataset. """

    data = table.lookup(text)                 # integers lookup-table

    return data


def extract_mikolov_sentences_data(ids, text, table, ocean_table):
    """ Function that extract labels and features, for the model, from dataset. """

    ocean_dict_file_path = "Dataset/Vocabulary/ocean_dict.txt"
    vocab_size = 60000

    file = tf.read_file(ocean_dict_file_path)
    lines = tf.string_split([file], '\n').values

    value = get_ocean_value(lines)

    data = table.lookup(text)                 # integers lookup-table

    ocean = ocean_table.lookup(text)

    # Create ocean vector
    ocean = ocean + 1
    ocean_new_table = tf.nn.embedding_lookup(value, ocean)

    # Compute the mean of the non_zero ocean vector
    non_zero = tf.count_nonzero(ocean, 0, dtype=tf.float32)
    ocean_vector = tf.reduce_sum(ocean_new_table, 0) / non_zero

    return data, ocean_vector


def extract_binarized_mikolov_sentences_data(features, ocean_vector):
    """ Function that convert in binary the labels and return the dataset composed by binarized labels and features. """

    # If Value is grater than 0 -> 1 else 0
    bin_ocean_vector = tf.greater(ocean_vector, 0)

    return features, bin_ocean_vector


def extract_mikolov_words(data):
    """ Create Skip-gram Model
        (context, target)
        context=[words-to-the-left of the target, words-to-the-right of the target]  """

    words_data = tf.data.Dataset.from_tensor_slices(data)
    
    # features
    target = words_data.skip(1)
    
    # labels
    context_left = words_data
    context_right = words_data.skip(2)

    data1 = tf.data.Dataset.zip((target, context_left))
    data2 = tf.data.Dataset.zip((target, context_right))

    dataset = data1.concatenate(data2)

    return dataset

def extract_mikolov_4vec(text, table, ocean_table):
    """ Create Vector of the context of the adjective 
        (target -2, target -1, target +1, target +2) 
        Initially i do that considering all the words of the sentences as target.
        After this, i apply a filter."""

    ocean_dict_file_path = "Dataset/Vocabulary/ocean_dict.txt"
    vocab_size = 60000

    file = tf.read_file(ocean_dict_file_path)
    lines = tf.string_split([file], '\n').values
    
    # Create a variable.
    value = get_ocean_value(lines)

    ocean = ocean_table.lookup(text)

    # Create ocean vector
    ocean = ocean + 1
    ocean_values = tf.nn.embedding_lookup(value, ocean)
    ocean_values = tf.data.Dataset.from_tensor_slices(ocean_values)

    data = table.lookup(text)
    data = tf.data.Dataset.from_tensor_slices(data)
    
    # labels
    target_word = data.skip(2)
    target_ocean = ocean_values.skip(2)
    
    # features 
    context_1 = data
    context_2 = data.skip(1)
    context_3 = data.skip(3)
    context_4 = data.skip(4)

    data1 = tf.data.Dataset.zip((target_word, context_1))
    data2 = tf.data.Dataset.zip((target_word, context_2))
    data3 = tf.data.Dataset.zip((target_word, context_3))
    data4 = tf.data.Dataset.zip((target_word, context_4))

    data = data1.concatenate(data2).concatenate(data3).concatenate(data4)

    # data = data.map(lambda context_1, context_2, context_3, context_4: tf.stack([context_1, context_2, context_3, context_4]), num_parallel_calls=8)

    dataset = tf.data.Dataset.zip((data, target_ocean))

    return dataset

    