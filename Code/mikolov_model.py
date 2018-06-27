#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import TFRecord_dataset as exp_TFR
from mikolov_features import extract_mikolov_sentences_data
from mikolov_embedding_model import extract_embedding


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
        dataset = corpus.map(lambda ids, text: extract_mikolov_sentences_data(ids, text, table, ocean_table), num_parallel_calls=8)

        # Delete all the sentences without adjective
        # dataset = dataset.filter(lambda features, ocean_vector: tf.reduce_all(tf.is_finite(ocean_vector)))
        dataset = dataset.filter(lambda features, ocean_value: tf.reduce_all(tf.is_finite(ocean_value)))

        dataset = dataset.padded_batch(batch_size=30, padded_shapes=([-1], [5]))

        return dataset

    return input_fn


def create_conv_layer(net, filter, windows_size, embedding_size, activation, padding, name, is_training):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        net = tf.layers.conv2d(net,                        # input
                               filters=filter,
                               kernel_size=[windows_size, embedding_size],
                               use_bias=False,
                               padding=padding,
                               activation=None,
                               data_format='channels_first',
                               strides=2
                              )

        if activation is not None:
            net = tf.layers.batch_normalization(net, axis=-1, training=is_training, fused=True)
            net = activation(net)

    return net


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

    embed, nce_weights, nce_biases = extract_embedding(features, params, trainable=False)

    embedding_size = params["embedding_size"]

    # Neural Network
    net = embed
    net = tf.expand_dims(net, 1)

    # Apply Convolution filtering on input sequence.

    net = create_conv_layer(net, 100, 3, 3, tf.nn.relu, 'SAME', 'conv1', is_training)     # Add a ReLU for non linearity.

    # Max pooling across output of Convolution+Relu.
    net = tf.layers.max_pooling2d(net,
                                  pool_size=4,  # POOLING_WINDOW,
                                  strides=2,  # POOLING_STRIDE,
                                  padding='SAME',
                                  data_format='channels_first',
                                  name='mpol1')

    net = create_conv_layer(net, 75, 3, 63, tf.nn.relu, 'SAME', 'conv2', is_training)

    net = tf.reduce_mean(net, axis=2, keep_dims=True)
    #net = tf.reduce_mean(net, axis=3, keep_dims=True)
    
    net = create_conv_layer(net, 50, 1, 32, tf.nn.relu, 'VALID', 'conv3', is_training)

    #net = tf.reduce_mean(net, axis=3, keep_dims=True)

    # net = create_conv_layer(net, 25, 1, 16, tf.nn.relu, 'VALID', 'conv4', is_training)
    
    net = tf.squeeze(net, axis=[2, 3])

    # net = create_fully_connected_layer(net, 100, tf.nn.relu, 'fc1', is_training)
    # net = create_fully_connected_layer(net, 50, tf.nn.relu, 'fc2', is_training)
    # net = create_fully_connected_layer(net, 20, tf.nn.relu, 'fc3', is_training)

    # net = tf.reduce_mean(net, axis=1, keep_dims=True)

    net = create_fully_connected_layer(net, 5, None, 'fc1', is_training)

    # net = tf.squeeze(net, axis=[1])
    
    # Predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'ocean': net}

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # labels = tf.expand_dims(labels, -1)

    # Compute the NCE loss, using a sample of the negative labels each time.
    # loss = tf.losses
    loss = tf.losses.mean_squared_error(labels=labels, predictions=net)

    rmse = tf.metrics.root_mean_squared_error(labels, net)

    # mean of each ocean value
    mean_o = tf.metrics.mean(net[:, 0])

    mean_c = tf.metrics.mean(net[:, 1])

    mean_e = tf.metrics.mean(net[:, 2])

    mean_a = tf.metrics.mean(net[:, 3])

    mean_n = tf.metrics.mean(net[:, 4])

    # rmse for each ocean value
    rmse_o = tf.metrics.root_mean_squared_error(labels[:, 0], net[:, 0])

    rmse_c = tf.metrics.root_mean_squared_error(labels[:, 1], net[:, 1])

    rmse_e = tf.metrics.root_mean_squared_error(labels[:, 2], net[:, 2])

    rmse_a = tf.metrics.root_mean_squared_error(labels[:, 3], net[:, 3])

    rmse_n = tf.metrics.root_mean_squared_error(labels[:, 4], net[:, 4])

    metric_ops = {'rmse': rmse, 'mean_openness': mean_o, 'mean_conscientiousness': mean_c,
                  'mean_extraversion': rmse_o, 'mean_agreeableness': mean_a, 'mean_neuroticism': mean_n,
                  'rmse_openness': rmse_o, 'rmse_conscientiousness': rmse_c,
                  'rmse_extraversion': rmse_e, 'rmse_agreeableness': rmse_a, 'rmse_neuroticism': rmse_n
                  }

    tf.summary.scalar('rmse', rmse[1])  # Tensorboard

    tf.summary.scalar('mean_openness', mean_o[1])  # Tensorboard
    tf.summary.scalar('mean_conscientiousness', mean_c[1])  # Tensorboard
    tf.summary.scalar('mean_extraversion', mean_e[1])  # Tensorboard
    tf.summary.scalar('mean_agreeableness', mean_a[1])  # Tensorboard
    tf.summary.scalar('mean_neuroticism', mean_n[1])  # Tensorboard

    tf.summary.scalar('rmse_openness', rmse_o[1])  # Tensorboard
    tf.summary.scalar('rmse_conscientiousness', rmse_c[1])  # Tensorboard
    tf.summary.scalar('rmse_extraversion', rmse_e[1])  # Tensorboard
    tf.summary.scalar('rmse_agreeableness', rmse_a[1])  # Tensorboard
    tf.summary.scalar('rmse_neuroticism', rmse_n[1])  # Tensorboard

    # Evaluation
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metric_ops)

    # Training
    assert mode == tf.estimator.ModeKeys.TRAIN

    # Embedding_extraction
    # tf.train.init_from_checkpoint('Model/Mikolv/001/', {'embedding_extraction/': 'feature_extraction/'})

    tf.train.init_from_checkpoint('Model/Mikolov/010/', {'embedding_extraction/': 'embedding_extraction/'})

    #check_op = tf.add_check_numerics_ops()
    #with tf.control_dependencies([check_op]):

    # We use the SGD optimizer.
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.005)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
