#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import tensorflow as tf


def csv2dict(file_csv):
    with open(file_csv, mode='r') as infile:
        reader = csv.reader(infile)
        my_dict = {rows[0].lower():rows[1:6] for rows in reader}

    return my_dict


def csv2txt(file_csv):

    f = open("Dataset/Vocabulary/ocean_dict.txt", "w")

    with open(file_csv, mode='r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            if rows[0].lower() != 'adjective':
                f.write(rows[0].lower())
                f.write('\t')
                f.write(rows[1])
                f.write('\t')
                f.write(rows[2])
                f.write('\t')
                f.write(rows[3])
                f.write('\t')
                f.write(rows[4])
                f.write('\t')
                f.write(rows[5])
                f.write('\n')
    f.close()

    
def get_key(line):

    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    DEFAULTS = [[""], [0.0], [0.0], [0.0], [0.0], [0.0]]
    # Decode the line into its fields
    field = tf.decode_csv(line, DEFAULTS, field_delim='\t')

    return field[0]


def get_ocean_value(line):

    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    DEFAULTS = [[""], [0.0], [0.0], [0.0], [0.0], [0.0]]

    # Decode the line into its fields
    field = tf.decode_csv(line, DEFAULTS, field_delim='\t')

    asd = tf.stack(field[1:], axis=1)
    asd = tf.concat([[[0, 0, 0, 0, 0]], asd], axis=0)

    return asd


def get_field(line):

    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    DEFAULTS = [[""], [0.0], [0.0], [0.0], [0.0], [0.0]]

    # Decode the line into its fields
    field = tf.decode_csv(line, DEFAULTS, field_delim='\t')

    key = field[0]
    value = tf.stack(field[1:], 0)

    return key, value
