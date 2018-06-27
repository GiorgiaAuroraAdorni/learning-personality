#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import TFRecord_dataset as exp_TFR
from extract_features import extract_features


class InputInitHook(tf.train.SessionRunHook):
    pass
    # def begin(self):
    #     self.tables_initializer = tf.tables_initializer()
    #
    # def after_create_session(self, session, coord):
    #     session.run(self.tables_initializer)


def create_input_fn(filename):
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

        # Extract labels and features and generate dataset
        dataset = corpus.map(lambda ids, text: extract_features(ids, text, table, ocean_table), num_parallel_calls=8)

        # Delete all the sentences without adjective
        # dataset = dataset.filter(lambda features, ocean_vector: tf.reduce_all(tf.is_finite(ocean_vector)))
        dataset = dataset.filter(lambda features, ocean_value: tf.reduce_all(tf.is_finite(ocean_value)))

        dataset = dataset.batch(batch_size=500)

        return dataset

    return input_fn  #, init_hook


def create_fully_connected_layer(net, units, activation, name, is_training):
    net = tf.layers.dense(net,  # input
                          units=units,  # number of neurons
                          use_bias=False,
                          # activation=tf.nn.sigmoid, # activation function
                          activation=None,
                          name=name)

    if activation is not None:
        net = tf.layers.batch_normalization(net, axis=-1, training=is_training, fused=True)
        net = activation(net)

    return net


def model_fn(features, labels, mode, params):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # FIXME: mancano i params

    # Neural Network
    net = features['bow_vector']

    net = create_fully_connected_layer(net, 100, tf.nn.relu, 'layer1', is_training)

    net = create_fully_connected_layer(net, 50, tf.nn.relu, 'layer2', is_training)

    net = create_fully_connected_layer(net, 20, tf.nn.relu, 'layer3', is_training)

    net = create_fully_connected_layer(net, 5, None, 'layer4', is_training)
    # net = create_fully_connected_layer(net, 1, None, 'layer3', is_training)

    # Fixed labels shape
    #labels = tf.expand_dims(labels, axis=-1)

    # Predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'ocean': net}

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Evaluation
    loss = tf.losses.mean_squared_error(labels=labels, predictions=net)

    rmse = tf.metrics.root_mean_squared_error(labels, net)

    mean_o = tf.metrics.mean(net[:, 0])

    mean_c = tf.metrics.mean(net[:, 1])

    mean_e = tf.metrics.mean(net[:, 2])

    mean_a = tf.metrics.mean(net[:, 3])

    mean_n = tf.metrics.mean(net[:, 4])

    metric_ops = {'rmse': rmse, 'mean_openness': mean_o, 'mean_conscientiousness': mean_c,
                  'mean_extraversion': mean_e, 'mean_agreeableness': mean_a, 'mean_neuroticism': mean_n}

    # metric_ops = {'rmse': rmse}

    tf.summary.scalar('rmse', rmse[1])  # Tensorboard

    tf.summary.scalar('mean_openness', mean_o[1])  # Tensorboard
    tf.summary.scalar('mean_conscientiousness', mean_c[1])  # Tensorboard
    tf.summary.scalar('mean_extraversion', mean_e[1])  # Tensorboard
    tf.summary.scalar('mean_agreeableness', mean_a[1])  # Tensorboard
    tf.summary.scalar('mean_neuroticism', mean_n[1])  # Tensorboard

    # Evaluation
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss,
                                          eval_metric_ops=metric_ops)

    # Training
    assert mode == tf.estimator.ModeKeys.TRAIN

    # check_op = tf.add_check_numerics_ops()

    # with tf.control_dependencies([check_op]):
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.0001)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
