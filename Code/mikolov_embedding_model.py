#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import math
import TFRecord_dataset as exp_TFR
from mikolov_features import extract_mikolov_sentences
from mikolov_features import extract_mikolov_words
from mikolov_features import extract_mikolov_4vec



def create_input_fn(filename, is_training):
    # init_hook = InputInitHook()

    def input_fn():
        # Load and parse dataset
        dataset = tf.data.TFRecordDataset(filename, compression_type='GZIP')
        corpus = dataset.map(exp_TFR.decode, num_parallel_calls=8)

        corpus = corpus.shuffle(5000000)

        # Build the dictionary
        # Extract the top 60000 most common words to include in our embedding vector
        vocab_file_path = "Dataset/Vocabulary/vocabulary.txt"
        vocab_size = 60000

        # Gather together all the unique words and index them with a unique integer value
        # Loop through every word in the dataset and assign it to the unique integer word identified.
        # Any words not within the top 60000 most common words will be marked with "-1" and replace with "UNK" token

        # Load the dictionary populated by keys corresponding to each unique word
        table = tf.contrib.lookup.index_table_from_file(vocabulary_file=vocab_file_path,
                                                        vocab_size=vocab_size,
                                                        key_column_index=1,
                                                        delimiter=' ')

        # Create a reverse_table that allows us to look up a word based on its unique integer identifier,
        # rather than looking up the identifier based on the word.
        # reverse_table = tf.contrib.lookup.index_to_string_table_from_file(vocabulary_file=vocab_file_path,
        #                                                                   vocab_size=vocab_size,
        #                                                                   value_column_index=1,
        #                                                                   delimiter=' ')

        # Load ocean dictionary
        ocean_dict_file_path = "Dataset/Vocabulary/ocean_dict_filtered.txt"
        ocean_dict_size = 634  # 636 before (deleted 2 adjective)

        # Ocean lookup-table
        ocean_table = tf.contrib.lookup.index_table_from_file(vocabulary_file=ocean_dict_file_path,
                                                              vocab_size=ocean_dict_size,
                                                              key_column_index=0,
                                                              delimiter='\t')

        # Extract data and generate dataset
        #sentences_dataset = corpus.map(lambda ids, text: extract_mikolov_sentences(ids, text, table, ocean_table), num_parallel_calls=8)
        #word_dataset = sentences_dataset.flat_map(extract_mikolov_words)

        dataset = corpus.flat_map(lambda ids, text: extract_mikolov_4vec(text, table, ocean_table))

        if not is_training:
            dataset = dataset.filter(lambda data, target_ocean: tf.reduce_all(tf.is_finite(target_ocean)))

        dataset = dataset.map(lambda data, target_ocean: (data), num_parallel_calls=8)

        dataset = dataset.batch(batch_size=10000)

        return dataset

    return input_fn


def model_fn(features, labels, mode, params):
    embed, nce_weights, nce_biases = extract_embedding(features, params, trainable=True)

    # Predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'embedding': embed}

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    labels = tf.expand_dims(labels, -1)

    # Compute the NCE loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                         biases=nce_biases,
                                         labels=labels + 1,
                                         inputs=embed,
                                         num_sampled=params["num_sampled"],
                                         num_classes=params["vocabulary_size"]))

    # Evaluation
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)

    # Training
    assert mode == tf.estimator.ModeKeys.TRAIN

    check_op = tf.add_check_numerics_ops()
    with tf.control_dependencies([check_op]):

        # We use the SGD optimizer.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def extract_embedding(features, params, trainable):

    with tf.variable_scope('embedding_extraction', reuse=tf.AUTO_REUSE):

        embeddings = tf.Variable(tf.random_uniform([params["vocabulary_size"], params["embedding_size"]], -1.0, 1.0),
                                 name='embeddings', trainable=trainable)

        nce_weights = tf.Variable(tf.truncated_normal([params["vocabulary_size"], params["embedding_size"]],
                                                      stddev=1.0 / math.sqrt(params["embedding_size"])),
                                  name='nce_weights', trainable=trainable)

        nce_biases = tf.Variable(tf.zeros([params["vocabulary_size"]]), name='nce_biases', trainable=trainable)

        embed = tf.nn.embedding_lookup(embeddings, features + 1)

    return embed, nce_weights, nce_biases
