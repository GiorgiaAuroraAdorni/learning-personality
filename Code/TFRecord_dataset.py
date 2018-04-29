#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def write_tfrecords(id, texts, writer):
    """Write the dataset (composed by the tuple id-text_words) into the TFRecord file"""

    texts = [tf.compat.as_bytes(s) for s in texts]
    # Create a feature
    feature = {'id': _bytes_feature(id),
               'text': _bytes_feature(texts)}

    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())


def decode(serialized_example):
    """Parses text and id from the given `serialized_example`."""

    features = tf.parse_single_example(
        serialized_example,  # features = feature
        features={
            # 'text': tf.VarLenFeature(tf.string),
            # Features
            'text': tf.FixedLenSequenceFeature([], tf.string, allow_missing=True),
            'id': tf.FixedLenFeature([1], tf.string),
        }
    )
    
    text = features['text']
    ids = features['id']

    return ids, text











