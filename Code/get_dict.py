#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import basic_w2v as w2v
import TFRecord_dataset as exp_TFR

test_filename = "Dataset/TFRecords/train.tfrecords.gz"

dataset = tf.data.TFRecordDataset(test_filename, compression_type = 'GZIP')

# map takes a python function and applies it to every sample
dataset = dataset.map(exp_TFR.decode)

## Creo un iteratore a partire dal dataset
iterator = dataset.make_one_shot_iterator()

## Tensore simile a un placeholder
next_element = iterator.get_next()

n=0

file = open("Dataset/Vocabulary/words.txt", "w") 

with tf.Session() as sess:
    while True:
        try:
            id, text = sess.run(next_element)

            for word in text:
                file.write(word.decode("utf-8")) 
                file.write('\n')
            if (n % 500 == 0):
            	print(n)

            n += 1

        except tf.errors.OutOfRangeError:
            break
    
# sentence: 1243000 test

file.close() 