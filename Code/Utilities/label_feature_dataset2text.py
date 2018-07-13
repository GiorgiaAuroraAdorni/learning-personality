#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import TFRecord_dataset as exp_TFR
import numpy as np
from extract_features import extract_features

# Load and parse dataset
filename_train = "Dataset/TFRecords/train.tfrecords.gz"
filename_test = "Dataset/TFRecords/test.tfrecords.gz"
dataset = tf.data.TFRecordDataset(filename_train, compression_type='GZIP')
corpus = dataset.map(exp_TFR.decode)

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
ocean_dict_file_path = "Dataset/Vocabulary/ocean_dict.txt"
ocean_dict_size = 636

# Ocean lookup-table
ocean_table = tf.contrib.lookup.index_table_from_file(vocabulary_file=ocean_dict_file_path,
                                                      vocab_size=ocean_dict_size,
                                                      key_column_index=0,
                                                      delimiter='\t')

# Extract labels and features and generate dataset
dataset = corpus.map(lambda ids, text: extract_features(ids, text, table, ocean_table))

# dataset = dataset.batch(batch_size=100)

iterator = dataset.make_initializable_iterator()  # iterator
bow, ocean = iterator.get_next()             # tensor


it = corpus.make_one_shot_iterator()
id, text = it.get_next()
file = open("Model_Dataset/ocean_data.txt", "w")

ids = 0

with tf.Session() as sess:
    tf.tables_initializer().run()
    sess.run(iterator.initializer)
    
    while True:
        try:
            # bow_, ocean_ = (sess.run([bow, ocean]))
            #id_, text_, ocean_ = (sess.run([id, text, ocean]))
            text_, ocean_ = (sess.run([ text, ocean]))
            # id_ = (str(id_[0]))

            #file.write(id_)
            file.write('id:')
            file.write(str(ids))
            file.write('\t')
            # for word in text_:
            #     file.write(word.decode('utf-8'))
            #     file.write(' ')
            # file.write('\t')
            file.write(str(ocean_))
            file.write('\n')

            # count = 0
            # for elem in bow_:
            #     if elem == 1:
            #         count += 1

            # if count == 0:
            #     print('\n')
            #     print('ERRORE: nella sentence id = ', i, 'non ci sono 1')
            #     print(sess.run(text[1]))
            #

            # for elem in bow_:
            #     file.write(str(int(elem)))
            #     file.write(' ')
            #
            # file.write('\t')
            #
            # file.write(ocean_)
            # file.write('\n')
            #

            if ids % 500 == 0:
                print(ids)

            ids += 1

        except tf.errors.OutOfRangeError:
            break
#file.close()
