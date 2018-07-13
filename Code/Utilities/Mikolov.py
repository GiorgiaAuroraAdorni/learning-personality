import json
import numpy as np
import tensorflow as tf

#raw_sentences = []
#with open('review100.json') as f:
#    for line in f:
#        data = json.loads(line)
#        element = json.loads(line)['text'].lower().split('\n') # convert to lower case
#        [i.split('\n', )[0] for i in element]
#        element = element.split('.')
#        data.append(element)

#print(data)
#sentences = []     # a list of sentences where each sentence is a list of words.
#raw_sentences = element.split('.') # raw sentences is a list of sentences.
#for sentence in raw_sentences:
#    sentences.append(sentence.split())
#
#print(sentences)

## create a dictionary
#words = []
#for x in data:
#    for word in x.split():
#        if word != '.': # because we don't want to treat . as a word
#            words.append(word)
#        print(word)
#        print('\n')
#
#words = set(words) # so that all duplicate words are removed
#
#print(words)
#
#print(data)

def get_bbox(str):
    obj = json.loads(str.decode('utf-8'))
    return np.array([obj['text']], dtype='str')

def get_multiple_bboxes(str):
    return [[get_bbox(x) for x in str]]

raw = tf.placeholder(tf.string, [None])
[parsed] = tf.py_func(get_multiple_bboxes, [raw], [tf.string])

with open('review100.json') as f:
    my_data = np.array(f.readlines())

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(parsed, feed_dict={raw: my_data}))
    print(sess.run(tf.shape(parsed), feed_dict={raw: my_data}))


