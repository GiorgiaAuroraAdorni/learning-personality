#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv 

def parse_csv(file_csv):
    with open(file_csv, mode='r') as infile:
        reader = csv.reader(infile)
        mydict = {rows[0].lower():rows[1:6] for rows in reader}

    return mydict