# -*- coding: utf-8 -*-

# Created by junfeng on 1/22/16.

from __future__ import print_function
import codecs
import functools
import os.path

import itertools
import numpy as np
from sklearn.externals import joblib

np.random.seed(7)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils.np_utils import accuracy
from keras.models import Graph
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.datasets import imdb

from cws_config import *
from eval_test import EvalTest


def read_lines(filename):
    """
    read all lines from a file
    :param filename: the file path
    :return: list of words list
    """
    lines = []
    with codecs.open(filename, "rb", encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if not l:
                print(l, filename)
                continue
            l = l.split()
            lines.append(l)
    return lines


class TestLSTM(object):
    def __init__(self, train_path, test_path, gold_path, dict_path):
        self.train_path = train_path
        self.test_path = test_path
        self.gold_path = gold_path
        self.dict_path = dict_path
        self.train_lines = read_lines(train_path)
        self.test_lines = read_lines(test_path)

        self.char2index = {}
        self.max_length = 0
        self.total_words = 0
        # initialize char2index, max_length and total_words
        self.find_max_len()
        # convert data
        self.X_train, self.y_train, self.X_test = self.load_data()
        # build model
        self.model = self.build_model()

    def find_max_len(self):
        """
        find max_length and construct char2index map
        :return:
        """
        for line in itertools.chain(self.train_lines, self.test_lines):
            # line is list of words
            line = ''.join(line)

            # find max length
            if self.max_length < len(line):
                print(line)
                self.max_length = len(line)
            # character to index
            index = 0
            for c in line:
                if c not in self.char2index:
                    self.char2index[c] = index
                    index += 1
        self.total_words = len(self.char2index)
        print("max length of line: %d" % self.max_length)
        print("total character: %d" % self.total_words)

    def load_data(self):
        """
        process train test lines to numpy array
        :return: x_train, y, x_test
        """
        # train process
        print("processing train file ...")
        x_train = []
        y = []
        for line in self.train_lines:
            old_line = ''.join(line)
            n = len(old_line)
            # 1 indicates continue character
            # 0 indicates start character
            labels = [1 for _ in range(n)]
            index = 0
            for word in line:
                labels[index] = 0
                index += len(word)
            characters = [self.char2index[c] for c in old_line]
            x_train.append(characters)
            y.append(labels)
        x_train = sequence.pad_sequences(x_train, maxlen=self.max_length, padding="post")
        y = sequence.pad_sequences(y, maxlen=self.max_length, padding="post")
        print("x_train.shape: ", x_train.shape)
        print("y.shape: ", y.shape)

        # test process
        print("processing test file ...")
        x_test = []
        for line in self.test_lines:
            old_line = ''.join(line)
            characters = [self.char2index[c] for c in old_line]
            x_test.append(characters)
        x_test = sequence.pad_sequences(x_test, maxlen=self.max_length, padding="post")
        print("x_test.shape: ", x_test.shape)

        return x_train, y, x_test

    def build_model(self):
        print('Build model...')
        model = Graph()
        model.add_input(name='input', input_shape=(self.max_length,), dtype=int)
        model.add_node(Embedding(self.total_words, 128, input_length=self.max_length),
                       name='embedding', input='input')
        model.add_node(LSTM(64, return_sequences=True), name='forward', input='embedding')
        model.add_node(LSTM(64, go_backwards=True, return_sequences=True), name='backward', input='embedding')

        model.add_node(LSTM(64, go_backwards=True, return_sequences=True), name='forward1', input='backward')
        model.add_node(LSTM(64, go_backwards=True, return_sequences=True), name='backward1', input='forward')

        model.add_node(Dropout(0.5), name='forward_dropout', merge_mode='ave', inputs=['forward', 'forward1'])
        model.add_node(Dropout(0.5), name='backward_dropout', merge_mode='ave', inputs=['backward', 'backward1'])

        model.add_node(LSTM(32, return_sequences=True), name='forward2', input='forward_dropout')
        model.add_node(LSTM(32, go_backwards=True, return_sequences=True), name='backward2', input='backward_dropout')

        model.add_node(LSTM(1, activation='sigmoid', return_sequences=True), name='sigmoid', merge_mode='ave',
                       inputs=['forward2', 'backward2'])
        model.add_output(name='output', input='sigmoid')

        # reshape y_train
        y_shape = self.y_train.shape
        self.y_train = np.reshape(self.y_train, (y_shape[0], y_shape[1], 1))
        return model

    def compile(self):
        # try using different optimizers and different optimizer configs
        self.model.compile('adam', {'output': 'binary_crossentropy'})

    def train(self, batch_size=128, nb_epoch=10, callbacks=None):
        base_dir = "../weights"
        if not os.path.isdir(base_dir):
            os.mkdir(base_dir)
        weight_path = os.path.join(base_dir, "weights-{epoch:02d}.hdf5")
        eval_callback = EvalTest(self.test_lines, self.X_test, weight_path)

        callbacks = callbacks if callbacks else [eval_callback]
        print("callbacks:")
        print(callbacks)

        print('Train...')
        self.model.fit({'input': self.X_train, 'output': self.y_train},
                       batch_size=batch_size,
                       nb_epoch=nb_epoch,
                       callbacks=callbacks)

    def predict(self, batch_size=128):
        print("Predict...")
        y_test_p = self.model.predict({'input': self.X_test}, batch_size=batch_size)['output']
        y_test = np.array(y_test_p)
        y_test = np.round(y_test)
        return y_test


if __name__ == "__main__":
    BATCH_SIZE = 32
    tl_model = TestLSTM(TRAIN_FILE, TEST_FILE, GOLD_PATH, DICT_PATH)
    tl_model.compile()
    tl_model.train(batch_size=BATCH_SIZE)

