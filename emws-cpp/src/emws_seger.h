//
// Created by junfeng on 3/2/16.
//

#ifndef PROJECT_EMWS_SEGER_H
#define PROJECT_EMWS_SEGER_H

#include <tuple>
#include <vector>
#include <map>

#include <armadillo>

#include <cereal/access.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/string.hpp>

#include "rapidjson/document.h"

#include "easylogging++.h"

#include "base_seger.h"
#include "Vocab.h"
#include "score_ret.h"

class emws_seger : public base_seger {


public:
    emws_seger(rapidjson::Document const &config);
    emws_seger(std::string const &model_path);

    virtual void train() override;

    virtual std::vector<std::u32string> predict(std::u32string const &sentence) const override;

    virtual bool save(std::string const &model_path) const override;

    virtual ~emws_seger() {}

private:
    void build_vocab(std::vector<std::vector<std::u32string> > const &sentences);
    std::map<std::u32string, Vocab> _vocab_from_new(std::vector<std::vector<std::u32string> > const &sentences);
    void reset_weights();
    void do_train(std::vector<std::vector<std::u32string> > const &sentences,
                  unsigned chunksize, unsigned current_iter);
    std::tuple<unsigned, double> train_gold_per_sentence(std::vector<std::u32string> const &sentence,
                                                         double learning_rate);

    std::tuple<arma::vec, arma::uvec, arma::uvec, arma::rowvec, arma::mat>
    predict_single_position(std::vector<std::u32string> &sent, unsigned pos,
                            unsigned prev2_label, unsigned prev_label,
                            std::vector<unsigned> states=std::vector<unsigned >()) const;

    std::tuple<arma::rowvec, arma::uvec>
    gen_feature(std::vector<std::u32string> &sent, unsigned pos,
                unsigned prev2_label, unsigned prev_label,
                unsigned future_label, unsigned future2_label) const;


    std::array<std::u32string, 9> gen_unigram_bigram(std::vector<std::u32string> &sent, unsigned pos) const;

    arma::uvec words2indices(std::vector<std::u32string> const &feat_vec) const;

    // eval functions
    score_ret eval(std::vector<std::u32string> const &sentences, std::string const &gold_path) const;

    std::vector<std::u32string> predict_sentence_greedy(std::u32string const &sentence) const;

    // serialization
    friend cereal::access;
    template <class Archive> void serialize(Archive & ar) {
        ar(
                size, alpha, min_alpha, min_count, seed, workers, iter, use_gold, train_path,
                test_raw_path, test_path, dev_path, quick_test, dict_path,
                score_script_path, pre_train, uni_path, bi_path, hybrid_pred,
                no_action_feature, no_bigram_feature, no_unigram_feature,
                no_binary_action_feature, no_sb_state_feature, no_right_action_feature,
                l2_rate, drop_out, binary_pred,
                train_corpus, test_corpus, dev_corpus, quick_test_corpus,
                mask, f_factor, f_factor2, non_fixed_param, pred_size, dropout_rate,
                dropout_size, train_mode, dev_test_result, epoch,
                vocab, index2word, hybrid_threshold, total_words
        );
        // syn0, syn1neg need handle separately
    }


private:
    rapidjson::Document config;
    // required fields
    unsigned size;
    double alpha;
    double min_alpha;
    unsigned min_count;
    unsigned seed;
    unsigned workers;
    unsigned iter;
    bool use_gold;
    std::string train_path;

    std::string test_raw_path;
    std::string test_path;
    std::string dev_path;
    std::string quick_test;
    std::string dict_path;

    std::string score_script_path;
    bool pre_train;
    std::string uni_path;
    std::string bi_path;
    bool hybrid_pred;

    bool no_action_feature;
    bool no_bigram_feature;
    bool no_unigram_feature;

    bool no_binary_action_feature;
    bool no_sb_state_feature;
    bool no_right_action_feature;

    double l2_rate;
    bool drop_out;
    bool binary_pred;

    std::vector<std::vector<std::u32string> > train_corpus;
    std::vector<std::u32string> test_corpus;
    std::vector<std::u32string> dev_corpus;
    std::vector<std::u32string> quick_test_corpus;

    unsigned mask;
    unsigned f_factor;
    unsigned f_factor2;
    unsigned non_fixed_param;
    unsigned pred_size;
    double dropout_rate;
    unsigned dropout_size;
    bool train_mode;
    std::vector<std::string> dev_test_result;
    unsigned epoch;

    // fields created from normal methods
    std::map<std::u32string, Vocab> vocab;
    std::vector<std::u32string> index2word;
    unsigned hybrid_threshold;
    unsigned total_words;
    arma::mat syn0;
    arma::mat syn1neg;

    // static member variables
    static el::Logger *logger;
    static std::u32string const START;
    static std::u32string const END;
    static std::u32string const label0_as_vocab;
    static std::u32string const label1_as_vocab;
    static std::u32string const unknown_as_vocab;

    //prefix for unigram/bigram state; no prefix for *char* unigram/bigrams
    static std::u32string const su_prefix;
    static std::u32string const sb_prefix;
    static std::array<std::u32string, 2> const state_varient;

};

#endif //PROJECT_EMWS_SEGER_H
