//
// Created by junfeng on 3/2/16.
//

#ifndef PROJECT_EMWS_SEGER_H
#define PROJECT_EMWS_SEGER_H

#include <tuple>
#include <vector>

#include "rapidjson/document.h"
#include "easylogging++.h"

#include "base_seger.h"

class emws_seger : public base_seger {


public:
    emws_seger(rapidjson::Document const &config);
    emws_seger(std::string const &model_path);

    virtual void train() override;

    virtual std::vector<std::u32string> predict(std::u32string const &sentence) const override;

    virtual bool save(std::string const &model_path) const override;

    virtual ~emws_seger() {}

private:
    rapidjson::Document config;
    // required fields
    unsigned size;
    double alpha;
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

    std::vector<std::u32string> train_corpus;
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
    bool train_model;
    std::vector<std::string> dev_test_result;
    unsigned epoch;

    // static member variables
    static el::Logger *logger;
    static std::string const START;
    static std::string const END;
    static std::string const label0_as_vocab;
    static std::string const label1_as_vocab;
    static std::string const unknown_as_vocab;

    //prefix for unigram/bigram state; no prefix for *char* unigram/bigrams
    static std::string const su_prefix;
    static std::string const sb_prefix;
    static std::tuple<char, char> const state_varient;

};

#endif //PROJECT_EMWS_SEGER_H
