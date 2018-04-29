#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import TFRecord_dataset as exp_TFR
from extract_features import extract_features

# Load and parse test dataset
test_filename = "Dataset/TFRecords/test.tfrecords.gz"
dataset = tf.data.TFRecordDataset(test_filename, compression_type='GZIP')

corpus = dataset.map(exp_TFR.decode)

# Load and parse training dataset
train_filename = "Dataset/TFRecords/train.tfrecords.gz"

# corpus_iterator = corpus.make_one_shot_iterator()  # iterator
# ids, text = corpus_iterator.get_next()             # tensor


test_dataset = corpus.map(extract_features)

iterator = test_dataset.make_initializable_iterator()
vector, ocean_vector = iterator.get_next()

tf.summary.scalar('Openness', ocean_vector[0])
tf.summary.scalar('Conscientiousness', ocean_vector[1])
tf.summary.scalar('Extraversion', ocean_vector[2])
tf.summary.scalar('Agreeableness', ocean_vector[3])
tf.summary.scalar('Neuroticism', ocean_vector[4])

summary = tf.summary.merge_all()
writer = tf.summary.FileWriter('Summary/bag_of_words_002')

with writer:
    with tf.Session() as sess:
        tf.tables_initializer().run()
        tf.global_variables_initializer().run()

        sess.run(iterator.initializer)

        writer.add_graph(sess.graph)

        i = 0
        while True:
            try:
                vector_, ocean_vector_, summary_ = sess.run([vector, ocean_vector, summary])
                print(vector_, ocean_vector_)

                writer.add_summary(summary_, i)

                i += 1

            except tf.errors.OutOfRangeError:
                break

# 3: Function to generate a training batch for the skip-gram model.

# 4: Build and train a skip-gram model.
# 5: Begin training.

# Our method integrates GloVe features with Gaussian Process regression as the learning algorithm.

# 6: Visualize the embeddings.
