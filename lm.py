#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Zimeng Qiu <zimengq@andrew.cmu.edu>

"""
F19 11-411/611 NLP Assignment 3 Task 1
N-gram Language Model Implementation Script
Zimeng Qiu Sep 2019

This is a simple implementation of N-gram language model

Write your own implementation in this file!
"""

import argparse

from utils import *
from collections import Counter
import string
import math
import time
from copy import deepcopy
from collections import defaultdict

### NOTE ###
# In this implementation, variable "word" actually means tokens in data, which include words, punctuations, etc

class LanguageModel(object):
    """
    Base class for all language models
    """
    def __init__(self, corpus, ngram, min_freq, uniform=False):
        """
        Initialize language model
        :param corpus: input text corpus to build LM on
        :param ngram: number of n-gram, e.g. 1, 2, 3, ...
        :param min_freq: minimum frequency threshold to set a word to UNK placeholder
                         set to 1 to not use this threshold
        :param uniform: boolean flag, set to True to indicate this model is a simple uniform LM
                        otherwise will be an N-gram model
        """
        # self.corpus: the original text
        self.corpus = corpus
        self.ngram = ngram
        self.min_freq = min_freq
        self.uniform = uniform

        
        self.flatten_text()
        self.least_freq_to_UNK()
        self.build()
        self.most_common_words(50)


    # takes in self.beginEnd
    # gives out self.beginEndFlat, which change the list list of 50, to list of 1
    def flatten_text(self):

        self.flat = []
        for line in self.corpus:
            for word in line:
                self.flat.append(word)


    # takes in self.beginEndFlat
    # gives out self.UNKreplaced, change any word that appears less than self.min_freq to 'UNK'
    def least_freq_to_UNK(self):
        freqCount = Counter(self.flat)
        self.UNK = deepcopy(self.corpus)
        #print(self.UNK)
        for line in self.UNK:
            for word in line:
                pos = line.index(word)
                if freqCount[word] < self.min_freq:
                    line[pos] = 'UNK'

    # takes in self.corpus
    # gives out self.beginEnd, with <s> and </s> properly added
    def add_beginAndEnd(self):

        # for each of the 50 lists, add <s> to front, add </s> to end
        for line in self.inputText:
            # add '</s>', '<s>' to the front
            line.insert(0,'<s>')
            line.insert(0,'</s>')



    def build(self):
        """
        Build LM from text corpus
        """

        self.ngramList = []
        self.commonList = []
        self.inputText = deepcopy(self.UNK)
    
        
        # build the list for uniform and unigram model
        if (self.ngram == 1):
            self.add_beginAndEnd()
            for line in self.inputText:
                for word in line[2:]:
                    self.ngramList.append(word)
                    self.commonList.append(word)
            if self.uniform == True:
                self.ngramList = set(self.ngramList)
                self.commonList = set(self.commonList)


        # build the list for bigram model
        if (self.ngram == 2):
            self.add_beginAndEnd()
            for line in self.inputText:
                for word in line[2:]:
                    pos = line.index(word)
                    # single words, i
                    self.ngramList.append(line[pos-1])
                    # two words tuple, i and i-1
                    self.ngramList.append(tuple([line[pos-1], word]))
    
                for word in line[3:]:
                    pos = line.index(word)
                    self.commonList.append((line[pos-1])+" "+word)


        # build the list for trigram model
        if (self.ngram == 3):
            self.add_beginAndEnd()
            for line in self.inputText:
                for word in line[2:]:
                    pos = line.index(word)
                    # two words, i-1, i-2
                    self.ngramList.append(tuple([line[pos-2], line[pos-1]]))
                    # three words tuple, i, i-1, i-2
                    self.ngramList.append(tuple([line[pos-2], line[pos-1], word]))

                for word in line[4:]:
                    pos = line.index(word)
                    self.commonList.append((line[pos-2])+" "+(line[pos-1])+" "+word)
        
        # print(self.commonList)
        self.ngramDict = Counter(self.ngramList)
        # print(len(self.ngramDict))
        self.commonDict = Counter(self.commonList)
    


    def most_common_words(self, k):
        """
        This function will only be called after the language model has been built
        Your return should be sorted in descending order of frequency
        Sort according to ascending alphabet order when multiple words have same frequency
        :return: list[tuple(token, freq)] of top k most common tokens
        """

        # print("tuples: ", self.commonDict.items())
        
        d = defaultdict(int)
        for line in self.UNK:
            if self.ngram == 1:
                for word in line:
                    if self.uniform == True:
                        d[word] = 1
                    else:
                        d[word] += 1
            if self.ngram == 2:
                for pair in zip(line[:-1],line[1:]):
                    d['{} {}'.format(pair[0],pair[1])] += 1
            if self.ngram == 3:
                for tri in zip(line[:-2],line[1:-1],line[2:]):
                    d['{} {} {}'.format(tri[0],tri[1],tri[2])] += 1
        t = sorted(d.items(), key=lambda x: x[0])
        l = sorted(t, key=lambda kv: kv[1],reverse=True)
        
        return l[:k]
    

def calculate_perplexity(models, coefs, data):
    """
    Calculate perplexity with given model
    :param models: language models
    :param coefs: coefficients
    :param data: test data
    :return: perplexity
    """
    # number of distinct vocabulary, for bigram and trigram smoothing
    vocab_count = len(models[0].ngramList)

    test_size = 0
    for line in data:
        test_size += len(line)

    # change any word that is not in the vocabulary to 'UNK'
    trainSet = set(models[0].ngramList)
    for line in data:
        for word in line:
            if word not in trainSet:
                pos = line.index(word)
                line[pos] = 'UNK'


    for line in data:
        line.insert(0, '<s>')
        line.insert(0, '</s>')
    

    # perplexity tracker, for adding each logged wordPerp, and for performing further calculations
    perplexity = 0

    # for each word (token) in test data
    for line in data:
        for word in line[2:]:
            # use each ngram model
            wordPerp = 0
            for model in models:
                if model.ngram == 1:
                    # Uniform Model
                    if model.uniform == True:
                        wordPerp += (coefs[0] * math.log(1/(test_size/3) + 1))
        
                    # Unigram Model
                    else:
                        count_i = model.ngramDict[word]
                        total_count = len(model.ngramList)
                        wordPerp += (coefs[1] * ((count_i + 1)/((test_size*10) + vocab_count)))


                # Bigram Model
                elif model.ngram == 2:
                    pos = line.index(word)
                    count_i_prev = model.ngramDict[(line[pos-1], word)] + 1
                    count_prev = model.ngramDict[line[pos-1]] + vocab_count
                    wordPerp += (coefs[2] * (count_i_prev/count_prev))


                # Trigram Model
                elif model.ngram == 3:
                    pos = line.index(word)
                    count_i_prev_prev = model.ngramDict[(line[pos-2], line[pos-1], word)] + 1
                    count_prev_prev = model.ngramDict[(line[pos-2], line[pos-1])] + vocab_count
                    wordPerp += (coefs[3] * (count_i_prev_prev/count_prev_prev*50))
            

            wordPerp = math.log(wordPerp)
            perplexity += wordPerp

    perplexity *= -1
    perplexity /= test_size
    perplexity = math.pow(2, perplexity)
                        
    return perplexity


# Do not modify this function!
def parse_args():
    """
    Parse input positional arguments from command line
    :return: args - parsed arguments
    """
    parser = argparse.ArgumentParser('N-gram Language Model')
    parser.add_argument('coef_unif', help='coefficient for the uniform model.', type=float)
    parser.add_argument('coef_uni', help='coefficient for the unigram model.', type=float)
    parser.add_argument('coef_bi', help='coefficient for the bigram model.', type=float)
    parser.add_argument('coef_tri', help='coefficient for the trigram model.', type=float)
    parser.add_argument('min_freq', type=int,
                        help='minimum frequency threshold for substitute '
                             'with UNK token, set to 1 for not use this threshold')
    parser.add_argument('testfile', help='test text file.')
    parser.add_argument('trainfile', help='training text file.', nargs='+')
    args = parser.parse_args()
    return args


# Main executable script provided for your convenience
# Not executed on autograder, so do what you want
if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    # load and preprocess train and test data
    train = preprocess(load_dataset(args.trainfile))
    test = preprocess(read_file(args.testfile))

    # build language models
    print("111111")
    uniform = LanguageModel(train, ngram=1, min_freq=args.min_freq, uniform=True)
    print("222222")
    unigram = LanguageModel(train, ngram=1, min_freq=args.min_freq)
    print("333333")
    bigram = LanguageModel(train, ngram=2, min_freq=args.min_freq)
    print("444444")
    trigram = LanguageModel(train, ngram=3, min_freq=args.min_freq)

    # calculate perplexity on test file
    ppl = calculate_perplexity(
        models=[uniform, unigram, bigram, trigram],
        coefs=[args.coef_unif, args.coef_uni, args.coef_bi, args.coef_tri],
        data=test)

    print("Perplexity: {}".format(ppl))




