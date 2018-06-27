#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tfmpl
import tensorflow as tf
import TFRecord_dataset as exp_TFR
from mikolov_features import extract_binarized_mikolov_sentences_data as extract_bin_data
from mikolov_features import extract_mikolov_sentences_data as extract_sent_data
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
        dataset = corpus.map(lambda ids, text: extract_sent_data(ids, text, table, ocean_table), num_parallel_calls=8)

        # Delete all the sentences without adjective
        # dataset = dataset.filter(lambda features, ocean_vector: tf.reduce_all(tf.is_finite(ocean_vector)))
        dataset = dataset.filter(lambda features, ocean_value: tf.reduce_all(tf.is_finite(ocean_value)))

        # Binarize the ocean vector
        dataset = dataset.map(extract_bin_data, num_parallel_calls=8)

        # Create batch 
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


@tfmpl.figure_tensor
def draw_confusion_matrix(matrix):
    '''Draw confusion matrix for MNIST.'''
    fig = tfmpl.create_figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    ax.set_title('Confusion matrix for big5 classification')
    
    tfmpl.plots.confusion_matrix.draw(
        ax, matrix,
        axis_labels=['0', '1'],
        normalize=True
    )

    return fig


def model_fn(features, labels, mode, params):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    embed, nce_weights, nce_biases = extract_embedding(features, params, trainable=False)

    embedding_size = params["embedding_size"]

    # Neural Network
    net = embed
    net = tf.expand_dims(net, 1)

    # Apply Convolution filtering on input sequence.

    net = create_conv_layer(net, 100, 7, 5, tf.nn.relu, 'SAME', 'conv1', is_training)     # Add a ReLU for non linearity.

    # Max pooling across output of Convolution+Relu.
    net = tf.layers.max_pooling2d(net,
                                  pool_size=4,  # POOLING_WINDOW,
                                  strides=2,  # POOLING_STRIDE,
                                  padding='SAME',
                                  data_format='channels_first',
                                  name='mpol1')

    net = create_conv_layer(net, 75, 5, 63, tf.nn.relu, 'SAME', 'conv2', is_training)
    
    net = create_conv_layer(net, 50, 3, 32, tf.nn.relu, 'SAME', 'conv3', is_training)

    net = tf.reduce_mean(net, axis=2, keepdims=True)

   	# net = tf.reduce_mean(net, axis=3, keepdims=True)

    net = create_conv_layer(net, 25, 1, 16, tf.nn.relu, 'VALID', 'conv4', is_training)

    net = tf.squeeze(net, axis=[2, 3])

    # net = create_fully_connected_layer(net, 100, tf.nn.relu, 'fc1', is_training)
    # net = create_fully_connected_layer(net, 50, tf.nn.relu, 'fc2', is_training)
    # net = create_fully_connected_layer(net, 20, tf.nn.relu, 'fc3', is_training)

    # net = tf.reduce_mean(net, axis=1, keepdims=True)

    net = create_fully_connected_layer(net, 5*2, None, 'fc1', is_training)

    net = tf.reshape(net, [-1, 5, 2])

    #net = tf.squeeze(net, axis=1)
    
    # Predictions
    bin_ocean = tf.argmax(net, axis=2)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'ocean': bin_ocean}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # labels = tf.expand_dims(labels, -1)

    # Compute the NCE loss, using a sample of the negative labels each time.
    labels = tf.cast(labels, dtype=tf.int32)

    loss = 0
    for i in range(5):
        loss += tf.losses.sparse_softmax_cross_entropy(labels[:, i], net[:, i, :])     

    # mean of each ocean value
    accuracy_o = tf.metrics.accuracy(labels[:, 0], bin_ocean[:, 0])

    accuracy_c = tf.metrics.accuracy(labels[:, 1], bin_ocean[:, 1])

    accuracy_e = tf.metrics.accuracy(labels[:, 2], bin_ocean[:, 2])

    accuracy_a = tf.metrics.accuracy(labels[:, 3], bin_ocean[:, 3])

    accuracy_n = tf.metrics.accuracy(labels[:, 4], bin_ocean[:, 4])

    metric_ops = {'accuracy_openness': accuracy_o, 'accuracy_conscientiousness': accuracy_c,
                  'accuracy_extraversion': accuracy_e, 'accuracy_agreeableness': accuracy_a, 
                  'accuracy_neuroticism': accuracy_n 
                  }

    for i, trait in enumerate(['O', 'C', 'E', 'A', 'N']):
        # Compute a per-batch confusion matrix
        c_matrix = tf.confusion_matrix(labels[:, i], bin_ocean[:, i], name='conf_matrix_batch_%s' % trait)
        
        # Create an accumulator variable to hold the counts
        confusion = tf.Variable(tf.zeros([2, 2], dtype=tf.int32), name='conf_matrix_acc_%s' % trait)
        
        # Create the update op for doing a "+=" accumulation on the batch
        confusion_update = confusion.assign_add(c_matrix)
        
        # Cast counts to float so tf.summary.image renormalizes to [0,255]
        # confusion_image = tf.reshape(tf.cast(confusion, tf.float32), [1, 2, 2, 1])

        # Get a image tensor for summary usage
        image_tensor = draw_confusion_matrix(confusion)
        
        # Rescale
        #confusion_image = (confusion_image - tf.reduce_min(confusion_image)) / (tf.reduce_max(confusion_image) - tf.reduce_min(confusion_image))

        c = tf.summary.image('confusion_%s' % trait, image_tensor)

        metric_ops['conf_mat_%s' % trait] = (c, confusion_update)

    tf.summary.scalar('accuracy_openness', accuracy_o[1])  # Tensorboard
    tf.summary.scalar('accuracy_conscientiousness', accuracy_c[1])  # Tensorboard
    tf.summary.scalar('accuracy_extraversion', accuracy_e[1])  # Tensorboard
    tf.summary.scalar('accuracy_agreeableness', accuracy_a[1])  # Tensorboard
    tf.summary.scalar('accuracy_neuroticism', accuracy_n[1])  # Tensorboard

    
    # Evaluation
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metric_ops)

    # Training
    assert mode == tf.estimator.ModeKeys.TRAIN

    # Embedding_extraction
    # tf.train.init_from_checkpoint('Model/Mikolv/001/', {'embedding_extraction/': 'feature_extraction/'})

    tf.train.init_from_checkpoint('Model/Mikolov/002/', {'embedding_extraction/': 'embedding_extraction/'})

    #check_op = tf.add_check_numerics_ops()
    #with tf.control_dependencies([check_op]):

    # We use the SGD optimizer.
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.0005)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
