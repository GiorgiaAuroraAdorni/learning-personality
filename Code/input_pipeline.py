#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import nltk
import numpy as np
import tensorflow as tf

def parse_function(input_line):
	input_line = input_line.decode('utf-8')
	return json.loads(input_line.lower())['text'].encode('utf-8')

# pre-trained Punkt tokenizer for English
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('crubadan')
#sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

from nltk.tokenize import sent_tokenize, word_tokenize

def tokenize_sentences(input_line):
	input_line = input_line.decode('utf-8')
	sentences = sent_tokenize(input_line.strip())
	#return np.array(sent_detector.tokenize(input_line.strip()), dtype='object')
	return np.array(sentences, dtype='object')

from nltk.corpus import stopwords
from nltk.classify import textcat

stop_words = set(stopwords.words('english')) 	# About 150 stopwords

def remove_stopwords(word_list):
    processed_word_list = []
    for word in word_list:
        word = textcat.TextCat().remove_punctuation(word)
        #word = word.lower() # in case they arenet all lower cased
        if word and word not in stop_words:
            processed_word_list.append(word)
    return processed_word_list


def tokenize_words(sentence):
	words = word_tokenize(sentence.decode('utf-8'))

	# words = [w for w in words if not w in stop_words]
	
	return np.array(remove_stopwords(words), dtype='object')
#	return np.array(remove_stopwords(input_line), dtype='object')

#filename = "review.json.gz"
#dataset = tf.data.TextLineDataset(filename, compression_type =  "GZIP")

filename = "review10.json"
dataset = tf.data.TextLineDataset(filename)

# Parso il file json estraendo solo la recensione 
texts = dataset.map(lambda input_line:
						tf.py_func(parse_function, [input_line], tf.string))

# Split upon punctuations to generate sentences
sentences = texts.flat_map(lambda input_line:
						tf.data.Dataset.from_tensor_slices(tf.py_func(tokenize_sentences, [input_line], tf.string)))

# Split to generate words
words = sentences.map(lambda input_line:
						tf.py_func(tokenize_words, [input_line], tf.string))

#dataset = dataset.shuffle(buffer_size = 10000)
#dataset = dataset.batch(5)

# Creo un iteratore a partire dal dataset
texts_iterator = texts.make_one_shot_iterator()

sentences_iterator = sentences.make_one_shot_iterator()

words_iterator = words.make_one_shot_iterator()

text_element = texts_iterator.get_next() # Tensore simile a un placeholder
sentence_element = sentences_iterator.get_next() # Tensore simile a un placeholder
word_element = words_iterator.get_next() # Tensore simile a un placeholder

# Consumo il dataset
with tf.Session() as sess:
	while True:
		try:
			(sess.run(text_element))
			print(sess.run(sentence_element))
			print(sess.run(word_element))
		except tf.errors.OutOfRangeError:
			break

# divido le sentences (stemming) es. tempi verbali, nomi maschili, lower case
# We also removed numbers, non-English words, urls, and extraneous information contained in PDF versions



# carico il dataset degli aggettivi (aggettivo, vettore ocean)			Ground Truth Collection
# dataset.filter
			
			
# 2: Build the dictionary 






# 3: Function to generate a training batch for the skip-gram model.

# window size: 5

# 4: Build and train a skip-gram model.
# 5: Begin training.


# Our method integrates GloVe features with Gaussian Process regression as the learning algorithm.


# 6: Visualize the embeddings.