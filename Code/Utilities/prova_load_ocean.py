#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
#from tensorflow.python import debug as tf_debug

file_csv = "Dataset/adj_ocean.csv"

COLUMNS = ['Adjective', 'Openness',
           'Conscientiousness', 'Extraversion',
           'Agreeableness', 'Neuroticism']

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
FIELD_DEFAULTS = [[""], [0.0], [0.0], [0.0], [0.0], [0.0]]

def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, FIELD_DEFAULTS)

    # Pack the result into a dictionary
    features = dict(zip(COLUMNS, fields))

    # Separate the label from the features
    label = features.pop('Adjective')
    Openness = features.pop('Openness')
    Conscientiousness = features.pop('Conscientiousness')
    Extraversion = features.pop('Extraversion')
    Agreeableness = features.pop('Agreeableness')
    Neuroticism = features.pop('Neuroticism')

    dictionary = dict(zip(label, [Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism]))

    return features, label, dictionary

ds = tf.data.TextLineDataset(file_csv).skip(1) # skip over the first line of the file, which contains a header     
ds = ds.map(_parse_line)
label = ds.map(lambda features, label, dictionary: dictionary)

# Creo un iteratore a partire dal dataset
iterator = label.make_one_shot_iterator()

next_element = iterator.get_next() # Tensore simile a un placeholder

# Consumo il dataset
with tf.Session() as sess:
	#sess = tf_debug.LocalCLIDebugWrapperSession(sess)

	while True:	
		try:
			print(sess.run(next_element))
		except tf.errors.OutOfRangeError:
			break

		
		
