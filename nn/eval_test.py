# -*- coding: utf-8 -*-

# Created by junfeng on 1/25/16.

import codecs
import os
import datetime

import numpy as np
from keras.callbacks import ModelCheckpoint

from cws_config import *


def parse_evaluation_result(path_to_evaluation_result):
    f = codecs.open(path_to_evaluation_result, 'rU', 'utf-8')
    lines = f.readlines()
    d_str = lines[-1]
    last_line_tokens = d_str.split()
    # last_line_tokens=f.readlines()[-1].split()
    if last_line_tokens[0] == '###' and len(last_line_tokens) == 14:
        recall, precision, f_score, oov_rate, oov_recall, iv_recall = [float(i) if i != "--" else i for i in
                                                                       last_line_tokens[-6:]]
        return path_to_evaluation_result.split('/')[-1], f_score, oov_recall, iv_recall, recall, precision
    else:
        print('error! Format of the EVALUATION RESULT does not match the standard!')


def segment_by_labels(sentence, labels):
    """
    segment sentence by labels
    :param sentence:
    :param labels:
    :return: list of words
    """

    tokens = []
    sentence = ''.join(sentence)
    for pos, c in enumerate(sentence):
        if pos == 0:
            tokens.append(c)
        else:
            label = labels[pos]
            if label == 1:
                tokens[-1] += c
            else:
                tokens.append(c)
    return tokens


class EvalTest(ModelCheckpoint):
    """
    eval test data using perl score
    """

    def __init__(self, test_corpus, X_test, filepath):
        """
        :param test_corpus: list of sentences
        :param X_test: X_test array
        :param filepath:
        :return:
        """
        super(EvalTest, self).__init__(filepath, verbose=1)
        self.test_corpus = test_corpus
        self.X_test = X_test

    def on_epoch_end(self, epoch, logs={}):
        super(EvalTest, self).on_epoch_end(epoch, logs)
        m = self.model
        y_test_p = m.predict({'input': self.X_test})['output']
        y_test = np.array(y_test_p)
        y_test = np.round(y_test)
        print("y_test.shape: ", y_test.shape)

        self.eval(y_test, epoch)

    def eval(self, y_test, epoch):
        """
        using perl score evaluate segmentation performance of the corpus
        :param y_test:
        :param epoch:
        :return: f_score, oov_recall, iv_recall
        """

        print('eval with "score" at epoch %s' % epoch)

        segmented_corpus = self.segment_corpus(y_test)
        time_str = datetime.datetime.now().strftime('.%d.%H.%M')
        time_str = 'result.con.cat' + time_str
        tmp_path = '../working_data/' + time_str + '.tmp.seg'
        tmp_eval_path = tmp_path + ".eval"

        with codecs.open(tmp_path, 'w', 'utf-8') as f:
            f.write(
                    '\n'.join(map(lambda s: ' '.join(s), segmented_corpus))
            )

        os.system(
                "perl " + SCORE_SCRIPT + "  " + DICT_PATH + " " + GOLD_PATH + "  " + tmp_path + "  >" + tmp_eval_path
        )
        _, f_score, oov_recall, iv_recall, recall, precision = parse_evaluation_result(tmp_eval_path)
        os.system("rm " + tmp_eval_path)
        os.system("rm " + tmp_path)
        print('F-score=%s, OOV-Recall=%s, IV-recall=%s' % (f_score, oov_recall, iv_recall))

        return f_score, oov_recall, iv_recall

    def segment_corpus(self, y_test):
        """
        segment corpus(all test data)
        :param y_test: predicted labels by self.model
        :return: segmented corpus
        """

        seg_corpus = []
        for sent, labels in zip(self.test_corpus, y_test):
            seg_corpus.append(segment_by_labels(sent, labels))
        return seg_corpus


if __name__ == "__main__":
    pass
