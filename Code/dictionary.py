#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import TFRecord_dataset as exp_TFR

test_filename = "Dataset/TFRecords/train.tfrecords.gz"

dataset = tf.data.TFRecordDataset(test_filename, compression_type='GZIP')
dataset = dataset.map(exp_TFR.decode)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

file = open("Dataset/Vocabulary/words.txt", "w")

with tf.Session() as sess:
    while True:
        try:
            ids, text = sess.run(next_element)

            for word in text:
                word = word.decode('utf-8')
                file.write(word)
                file.write('\n')

        except tf.errors.OutOfRangeError:
            break

file.close()

# sentence: 1243000 test
# sentence: 4974000 train
