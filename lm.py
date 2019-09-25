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
from copy import deepcopy


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

        self.add_beginAndEnd_to_each_sentense()
        self.flatten_text()
        self.least_freq_to_UNK()
        # self.build()

        return

    # takes in self.corpus
    # gives out self.beginEnd, with <s> and </s> properly added
    def add_beginAndEnd_to_each_sentense(self):
        self.beginEnd = deepcopy(self.corpus)
        # for each of the 50 lists, add <s> to front, add </s> to end
        for line in self.beginEnd:
            # add'<s>' to front, add '</s>' to end
            line.insert(0,'<s>')
            line.append('</s>')

            # replace each period with two elements '</s>', '<s>'
            for word in line:
                if word == '.':
                    pos = line.index(word)
                    line[pos:pos+1] = ('</s>', '<s>')

            # get rid of the last two elements, '<s>', '</s>'
            if line[-1] == '</s>' and line[-2] =='<s>':
                line = line[:len(line)-2]

        return

    # takes in self.beginEnd
    # gives out self.beginEndFlat, which change the list list of 50, to list of 1
    def flatten_text(self):
        self.beginEndFlat = []
        for line in self.beginEnd:
            for word in line:
                self.beginEndFlat.append(word)

        return

    # takes in self.beginEndFlat
    # gives out self.UNKreplaced, change any word that appears less than self.min_freq to 'UNK'
    def least_freq_to_UNK(self):
        
        self.UNKreplaced = deepcopy(self.beginEndFlat)
        freqCount = Counter(self.UNKreplaced)
        for i in range(len(self.UNKreplaced)):
            if freqCount[self.UNKreplaced[i]] < self.min_freq:
                self.UNKreplaced[i] = 'UNK'
        return 

    """
    Build LM from text corpus
    """
    def build(self):
        
        # Change any word that appears less than self.min_freq to "UNK"
        
        ngramList = []

        # build the list for uniform and unigram model
        if (self.ngram == 1):
            ngramList = self.trimmedText

        # build the list for bigram model
        if (self.ngram == 2):
            ngramList = ngramList+self.trimmedText
            for i in range(len(self.trimmedText)-1):
                ngramList.append(tuple([self.trimmedText[i], self.trimmedText[i+1]]))

        # build the list for trigram model
        if (self.ngram == 3):
            ngramList = ngramList+self.trimmedText
            for i in range(len(self.trimmedText)-1):
                ngramList.append(tuple([self.trimmedText[i], self.trimmedText[i+1]]))
            for i in range(len(self.trimmedText)-2):
                ngramList.append(tuple([self.trimmedText[i], self.trimmedText[i+1], self.trimmedText[i+2]]))

        # print(ngramList)
        self.ngramDict = Counter(ngramList)
        # print(ngramDict)
        return ngramList

    def most_common_tokens(self, k):
        """
        This function will only be called after the language model has been built
        Your return should be sorted in descending order of frequency
        Sort according to ascending alphabet order when multiple words have same frequency
        :return: list[tuple(token, freq)] of top k most common tokens
        """
        # Write your own implementation here

        raise NotImplemented


def calculate_perplexity(models, coefs, data):
    """
    Calculate perplexity with given model
    :param models: language models
    :param coefs: coefficients
    :param data: test data
    :return: perplexity
    """
    # perplexity = 1
    # for model in models:
    #     if model.ngram = 1:
    #         if model.uniform:
    #             for word in data[0]:
    #                 perplexity *= 1/len(model.trimmedText)
    #             return perplexity
    #         else:
    #             freqCount = Counter(model)
    #             for word in data[0]:
    #                 perplexity *= freqCount[word]/len(model.trimmedText)
    #             return perplexity

    #     elif model.ngram = 2:
    #         return perplexity

    #     elif model.ngram = 3:
    #         return perplexity

    

    return 


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




