# -*- coding: utf-8 -*-
import sys, datetime, time, codecs
from seg2 import *
import numpy as np


def segment_sentence(model, sent):
    old_sentence = "".join(sent)
    sentence = map(full2halfwidth, old_sentence)  # all half-width version, used to predict label...
    prev2_label, prev_label = 0, 0
    scores = np.zeros(len(old_sentence))
    scores[0] = 0.0

    for pos, char in enumerate(old_sentence):  # char is still the char from original sentence, for correct eval
        if pos == 0:
            label = 0
        else:
            score_list, _, _, _, _ = model.predict_sigle_position(sentence, pos, prev2_label, prev_label)

            if model.binary_pred:
                score_list = score_list[:2]
            elif model.alter:
                old_char = old_sentence[pos]
                if old_char in model.vocab and model.vocab[old_char].count > model.hybrid_threshold:
                    score_list = score_list[-2:]
                else:
                    # score_list = score_list[:2]
                    x, y = score_list[:2], score_list[-2:]
                    score_list = [(x[i] + y[i]) / 2.0 for i in range(2)]
            elif model.hybrid_pred:
                x, y = score_list[:2], score_list[-2:]
                score_list = [(x[i] + y[i]) / 2.0 for i in range(2)]
            else:
                score_list = score_list[-2:]

            scores[pos] = score_list[1]
            # transform score to binary label
            if score_list[1] > 0.5:
                label = 1
            else:
                label = 0
        prev2_label = prev_label
        prev_label = label

    return scores


def bidirect_segment(forward_model, back_model, text_corpus):
    threshold = forward_model.hybrid_threshold

    tic = time.time()
    count = 0
    seg_corpus = []

    for sent_no, sent in enumerate(test_corpus):
        if sent_no % 100 == 0:
            print 'num of sentence segmented:', sent_no, '...'

        tokens = []
        if sent:
            forward_scores = segment_sentence(forward_model, sent)
            back_scores = segment_sentence(back_model, sent[::-1])
            back_scores = back_scores[::-1]

            # calculates scores
            scores = np.zeros_like(forward_scores)
            scores[0] = forward_scores[0]
            for i in range(1, len(forward_scores)):
                scores[i] = (forward_scores[i] + back_scores[i-1]) / 2

            for pos, score in enumerate(scores):
                if score > 0.5:
                    tokens[-1] += sent[pos]
                else:
                    tokens.append(sent[pos])
            count += len(sent)
        seg_corpus.append(tokens)

    diff = time.time() - tic
    print 'segmentation done!'
    print 'time spent:', diff, 'speed=', count / float(diff), 'characters per second'

    return seg_corpus


if __name__ == '__main__':
    print '\n\nScript for conducting segmentation with an existing model...'
    print '\nArg: 1. forward_model_path, 2. back_model_path 3. file_to_be_segmented,  4. path to output'

    forward_model_path, back_model_path, test_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    print '\nreading testing corpus...'
    test_corpus = [''.join(line.split()) for line in codecs.open(test_path, 'rU', 'utf-8')]

    print '\nloading forward model...'
    forward_model = Seger.load(forward_model_path)
    forward_model.drop_out = False
    forward_model.alter = True

    print '\nloading back model...'
    back_model = Seger.load(back_model_path)
    back_model.drop_out = False
    back_model.alter = True

    seged = bidirect_segment(forward_model, back_model, test_corpus)

    print '\nwriting segmented corpus to file', out_path

    with codecs.open(out_path, 'w', 'utf-8') as f:
        f.write(
                '\n'.join(map(lambda s: ' '.join(s), seged))
        )

    print 'written done!'
