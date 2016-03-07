//
// Created by junfeng on 3/7/16.
//

#ifndef ZH_SEGER_VOCAB_H
#define ZH_SEGER_VOCAB_H

struct Vocab {
    Vocab(unsigned c) : count(c) {}
    Vocab() = default;
    unsigned count;
    unsigned long index;
    double sample_probability = 1.0;

};

#endif //ZH_SEGER_VOCAB_H
