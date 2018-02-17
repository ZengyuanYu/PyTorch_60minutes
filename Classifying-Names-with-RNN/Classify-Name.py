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
#print(all_letters)
n_letters = len(all_letters)
#print(n_letters)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print('ASCII码转化',unicodeToAscii('Ślusàrski')) # unicodeToAscii 目的是将其他字母转化成为ascii中有的
print('/n')
# Build the category_lines dictionary, a list of names per language
category_lines = {}# dict
all_categories = []# list

# 将读取的文件显示成行
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]#category为文件夹名字
    all_categories.append(category)#遍历将类别append到all_categories的尾部
    lines = readLines(filename)#将txt名字一个文件夹里面排成一行
    category_lines[category] = lines
    #print(category,category_lines[category])# 查看字典中元素和key
n_categories = len(all_categories) #语言种类数目
print('共有语言：%s种' % n_categories)

# print(category_lines['German'][:5])#输出列表中的key
# print(all_categories[0])

import torch
# 找到letter在‘abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;'’的索引值，n_letter=57
def letterToIndex(letter):
    return all_letters.find(letter)

# 转换一个字母为<1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# 转换字符串为 <line_length x 1 x n_letters>,
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

print(letterToTensor('J'))

print(lineToTensor('Jones').size())
