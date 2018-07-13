#!/usr/bin/env python
# -*- coding: utf-8 -*-

f = open("Dataset/Vocabulary/ocean_dict.txt", "r")

lines = f.readlines()
f.close()

f = open("Dataset/Vocabulary/prova_ocean.txt", "w")

for line in lines:
    word = line.split('\t')
    f.write(word[0])
    f.write('\n')

f.close()
