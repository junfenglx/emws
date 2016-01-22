# -*- coding: utf-8 -*-

# Created by junfeng on 1/22/16.

from __future__ import print_function
import codecs

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

TRAIN_FILE = "../working_data/pku_train"
TEST_FILE = "../working_data/pku_test.raw"


def find_max_len():
    def iter_lines(filename, max_len=0):
        with codecs.open(filename, "rb", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                line = line.replace(' ', '')
                if max_len < len(line):
                    print(line)
                    max_len = len(line)
        return max_len

    max_length = iter_lines(TRAIN_FILE)
    max_length = iter_lines(TEST_FILE, max_len=max_length)
    return max_length


MAX_LEN = 1019


def load_data():
    def read_lines(filename):
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

    # train process
    print("processing train file ...")
    train_lines = read_lines(TRAIN_FILE)
    x_train = []
    y = []
    for line in train_lines:
        old_line = ''.join(line)
        n = len(old_line)
        # 1 indicates continue character
        # 0 indicates start character
        labels = [1 for _ in range(n)]
        index = 0
        for word in line:
            labels[index] = 0
            index += len(word)
        characters = [ord(c) for c in old_line]
        x_train.append(characters)
        y.append(labels)
    x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN, padding="post")
    y = sequence.pad_sequences(y, maxlen=MAX_LEN, padding="post")
    print("x_train.shape: ", x_train.shape)
    print("y.shape: ", y.shape)

    # test process
    print("processing test file ...")
    test_lines = read_lines(TEST_FILE)
    x_test = []
    for line in test_lines:
        old_line = ''.join(line)
        characters = [ord(c) for c in old_line]
        x_test.append(characters)
    x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN, padding="post")
    print("x_test.shape: ", x_test.shape)

    return x_train, y, x_test


"""
x_train.shape:  (19054, 1019)
y.shape:  (19054, 1019)

x_test.shape:  (1944, 1019)
"""

MAX_FEATURES = 20000
BATCH_SIZE = 32


def build_model():
    print('Build model...')
    model = Graph()
    model.add_input(name='input', input_shape=(MAX_LEN,), dtype=int)
    model.add_node(Embedding(MAX_FEATURES, 128, input_length=MAX_LEN),
                   name='embedding', input='input')
    model.add_node(LSTM(64), name='forward', input='embedding')
    model.add_node(LSTM(64, go_backwards=True), name='backward', input='embedding')
    model.add_node(Dropout(0.5), name='dropout', inputs=['forward', 'backward'])
    model.add_node(Dense(1, activation='sigmoid'), name='sigmoid', input='dropout')
    model.add_output(name='output', input='sigmoid')
    return model

if __name__ == "__main__":

    # print(find_max_len())

    # X_train, y_train, X_test = load_data()
    # joblib.dump(X_train, "../data/X_train.pkl")
    # joblib.dump(y_train, "../data/y_train.pkl")
    # joblib.dump(X_test, "../data/X_test.pkl")

    X_train_pkl = "../data/X_train.pkl"
    y_train_pkl = "../data/y_train.pkl"
    X_test_pkl = "../data/X_test.pkl"

    X_train = joblib.load(X_train_pkl)
    y_train = joblib.load(y_train_pkl)
    print("X_train.shape: ", X_train.shape)
    print("y_train.shape: ", y_train.shape)

    model = build_model()

    # try using different optimizers and different optimizer configs
    model.compile('adam', {'output': 'binary_crossentropy'})

    print('Train...')
    model.fit({'input': X_train, 'output': y_train},
              batch_size=BATCH_SIZE,
              nb_epoch=10)

    X_test = joblib.load(X_test_pkl)
    print("X_test.shape: ", X_test.shape)

    y_test_p = model.predict({'input': X_test}, batch_size=BATCH_SIZE)['output']
    y_test = np.array(y_test_p)
    y_test = np.round(y_test)
    print("y_test.shape: ", y_test.shape)
    joblib.dump(y_test, "../data/y_test.pkl")
