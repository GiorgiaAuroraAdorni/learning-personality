#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import TFRecord_dataset as exp_TFR
from extract_features import extract_features


def create_input_fn(filename):
    def input_fn():
        # Load and parse dataset
        dataset = tf.data.TFRecordDataset(filename, compression_type='GZIP')
        corpus = dataset.map(exp_TFR.decode)

        # Extract labels and features and generate dataset
        dataset = corpus.map(extract_features)

        return dataset
    
    return input_fn


def model_fn(features, labels, mode, params):
    # FIXME: mancano i params

    # Neural Network
    net = features['bow_vector']

    net = tf.layers.dense(net,                       # input
                          units=20,                  # number of neurons
                          activation=tf.nn.sigmoid,  # activation function
                          name='layer1')

    net = tf.layers.dense(net,              # input
                          units=5,          # number of neurons
                          activation=None,  # activation function (linear output)
                          name='layer2')

    # Predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'ocean': net}

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Evaluation
    loss = tf.losses.mean_squared_error(labels=labels,
                                        predictions=net)

    metric_ops = {'rmse': tf.sqrt(loss)}

    tf.summary.scalar('rmse', tf.sqrt(loss))     # Tensorboard

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metric_ops)

    # Training
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradDAOptimizer(learning_rate=0.9)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
