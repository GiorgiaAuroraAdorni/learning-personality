#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import json
import nltk
import numpy as np
import tensorflow as tf
from load_ocean import parse_csv
from nltk.corpus import stopwords
from nltk.classify import textcat
from nltk.tokenize import sent_tokenize, word_tokenize

def parse_function(input_line):
    input_line = input_line.decode('utf-8')
    corpus = json.loads(input_line.lower())['text'].encode('utf-8')
    corpus_id = json.loads(input_line)['review_id'].encode('utf-8')

    return corpus_id, corpus

def split_dataset(dataset, ratio, n):
    #dataset = dataset.shuffle(buffer_size = 30) 
    ## shuffle eseguito da linea di comando
    #time bash -c "zcat review.json.gz | shuf | gzip  > shuf_review.json.gz"

    count_train = (n * ratio) // 100
    train = dataset.take(count_train)
    test = dataset.skip(count_train)
    return train, test

def tokenize_sentences(input_line):
    input_line = input_line.decode('utf-8')
    sentences = sent_tokenize(input_line.strip())
    #return np.array(sent_detector.tokenize(input_line.strip()), dtype='object')
    sentences = np.array(sentences, dtype = 'object')
    return sentences

punctuaction = set(['.', ',', ';', ':', '?', '!', '..', '...', '-', '(', ')', '$', '/', '\\', '^'])
number = re.compile(r"^\d+$")
stop_words = set(stopwords.words('english')).union(punctuaction) # About 150 stopwords

def remove_stopwords(word_list):
    processed_word_list = []
    for word in word_list:
       # word = textcat.TextCat().remove_punctuation(word)
        #word = word.lower() # in case they arenet all lower cased
        if word and word not in stop_words and not number.match(word):
            processed_word_list.append(word)
    return processed_word_list

def tokenize_words(sentence):
    words = word_tokenize(sentence.decode('utf-8'))
    # words = [w for w in words if not w in stop_words]
    words = np.array(remove_stopwords(words), dtype = 'object')
    return words

def combine_for_words(id_line, input_line):
    sentences = tf.py_func(tokenize_sentences, [input_line], tf.string)

    id = tf.data.Dataset.from_tensors(id_line).repeat()
    ds = tf.data.Dataset.from_tensor_slices(sentences)

    return tf.data.Dataset.zip((id, ds))

# carico il dataset degli aggettivi (aggettivo, vettore ocean)          Ground Truth Collection
file_csv = "Dataset/ocean.csv"
ocean_dict = parse_csv(file_csv)

def adjective_in_sentence(word_list):
    for word in word_list:
        word = word.decode('utf-8')
        if word in ocean_dict.keys():
            return True
    return False

def text2words(texts):

    # Split upon punctuations to generate sentences
    sentences = texts.flat_map(combine_for_words)

    # Split to generate words
    words = sentences.map(lambda id_line, input_line:
                           (id_line, tf.py_func(tokenize_words, [input_line], tf.string)))

    words = words.filter(lambda id, word: 
                                    tf.py_func(adjective_in_sentence, [word], tf.bool))

    return words

