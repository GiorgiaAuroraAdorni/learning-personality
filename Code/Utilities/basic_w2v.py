#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import collections
# import math
# import os
# import sys
# import argparse
# import random
# from tempfile import gettempdir
# import zipfile

import numpy as np
import tensorflow as tf
from load_ocean import csv2dict

# Load adjective dataset (adjective, ocean vector)
file_csv = "Dataset/ocean.csv"
ocean_dict = csv2dict(file_csv)

data1 = [ -1, -1, -1, -1, 338, -1]
sentece1 = ['pleasantly', 'surprised', 'commonwealth', 'reading', 'negative', 'reviews']

data2 = [ -1, -1, -1, -1, 404, 208, -1]
sentece2 = ['excellent', 'ayce', 'eat', 'value', 'prompt', 'friendly', 'service']

new_data = list(filter(lambda x: x != -1, data1))

print(new_data)


ocean = []
for key in data1:
    if key != -1:
        element = ocean_dict.get("none", key)
        print(element)
        ocean.append(element)
print(ocean)


# ocean = []
# for word in sentece2:
#     if word in ocean_dict.keys():
#         element = ocean_dict.get(word, "none")
#         print(element)
#         ocean.append(element)
# print(ocean)


def get_batch_windowing(text, table, reverse_table):
    """Function to generate a training batch for the skip-gram model."""
    # ??

    data = table.lookup(text)
    # rev_text = reverse_table.lookup(data)
    ids = tf.data.Dataset.from_tensor_slices(data)
    # window_size = 2
    return tf.data.Dataset.zip((ids, ids.skip(1)))


def get_batch(text, table, reverse_table):
    """Function to generate a training batch for the skip-gram model."""
    # ??

    data = table.lookup(text)
    # rev_text = reverse_table.lookup(data)
    ids = tf.data.Dataset.from_tensor_slices(data)
    return ids


def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1

    return temp

    # corpus_ids = corpus.flat_map(lambda id, sentence: w2v.get_batch(sentence, table, reverse_table))
    # corpus_ids_iterator = corpus_ids.make_initializable_iterator()
    # ids = corpus_ids_iterator.get_next()

    # sess.run(corpus_ids_iterator.initializer)

    # word = sess.run(ids)
    # print(word)
    # print(sess.run(ids))
