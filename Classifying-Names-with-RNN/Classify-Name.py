#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by XiaoYu on 18-2-13
from __future__ import unicode_literals, print_function, division
from io import open
import glob

def findFiles(path): return glob.glob(path)

print(findFiles('data/names/*.txt'))

import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(unicodeToAscii('Ślusàrski'))#unicodeToAscii 目的是将其他字母转化成为ascii中有的

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]#category为文件夹名字
    all_categories.append(category)#遍历将类别append到all_categories的尾部
    lines = readLines(filename)#将txt名字一个文件夹里面排成一行
    category_lines[category] = lines
    print(category_lines)
n_categories = len(all_categories)

