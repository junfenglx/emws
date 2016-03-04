//
// Created by junfeng on 3/2/16.
//

#include <iostream>

#include "emws_seger.h"

emws_seger::emws_seger(rapidjson::Document const &config) {

    logger->info("### Initialization of the segmentation model ###");

    this->config.CopyFrom(config, this->config.GetAllocator());

    // fetch parameters from config;
    size = config["size"].GetUint();
    alpha = config["alpha"].GetDouble();
    min_count = config["min_count"].GetUint();
    seed = config["seed"].GetUint();
    workers = config["workers"].GetUint();
    iter = config["iter"].GetUint();
    use_gold = config["use_gold"].GetBool();
    // train_path = config["train_path"].GetString();

    // test_raw_path = config["test_raw_path"].GetString();
    // test_path = config["test_path"].GetString();
    // dev_path = config["dev_path"].GetString();
    // quick_test = config["quick_test"].GetString();
    // dict_path = config["dict_path"].GetString();

    // score_script_path = config["score_script_path"].GetString();
    pre_train = config["pre_train"].GetBool();
    // uni_path = config["uni_path"].GetString();
    // bi_path = config["bi_path"].GetString();
    hybrid_pred = config["hybrid_pred"].GetBool();

    no_action_feature = config["no_action_feature"].GetBool();
    no_bigram_feature = config["no_bigram_feature"].GetBool();
    no_unigram_feature = config["no_unigram_feature"].GetBool();

    no_binary_action_feature = config["no_binary_action_feature"].GetBool();
    no_sb_state_feature = config["no_sb_state_feature"].GetBool();
    no_right_action_feature = config["no_right_action_feature"].GetBool();

    // another parameters

    l2_rate = 0.001;// rate for L2 regularization

    drop_out = false;

    binary_pred = false;

    // TODO loading corpora

    mask = 15;
    if (no_right_action_feature) {
        mask -= 3;
        logger->info("mask: %v", mask);
    }
    else if (no_sb_state_feature) {
        mask -= 1;
        logger->info("mask: %v", mask);
    }

    if (no_action_feature) {
        mask -= 3;
        logger->info("mask: %v", mask);
    }
    else if (no_sb_state_feature) {
        mask -= 1;
        logger->info("mask: %v", mask);
    }

    if (no_unigram_feature) {
        mask -= 5;
        logger->info("mask: %v", mask);
    }
    if (no_bigram_feature) {
        mask -= 4;
        logger->info("mask: %v", mask);
    }

    f_factor = mask;
    f_factor2 = 2;

    if (no_binary_action_feature) {
        f_factor2 = 0;
        logger->info("f_factor2: %v", f_factor2);
    }

    non_fixed_param = f_factor * size;
    pred_size = non_fixed_param + f_factor2;

    if (drop_out) {
        dropout_rate = 0.5;
        dropout_size = (unsigned int) (dropout_rate * non_fixed_param);
        logger->info("using dropout_rate: %v, dropout_size=%v", dropout_rate, dropout_size);
    }

    logger->info("Learning rate: %v", alpha);
    logger->info("Feature (layer1) size: %v", size);
    logger->info("Predicate vec size: %v", pred_size);
    logger->info("f_factor: %v", f_factor);
    logger->info("f_factor2: %v", f_factor2);

    if (pre_train) {
        // TODO implements pre train
    }

    epoch = 0;

}

emws_seger::emws_seger(std::string const &model_path) {

}

void emws_seger::train() {
    logger->info("start train ...");
}

std::vector<std::u32string> emws_seger::predict(std::u32string const &sentence) const {
    return std::vector<std::u32string>();
}

bool emws_seger::save(std::string const &model_path) const {
    return false;
}

el::Logger *emws_seger::logger = el::Loggers::getLogger("emws_seger");

std::string const emws_seger::START = "#S#";

std::string const emws_seger::END = "#E#";

std::string const emws_seger::label0_as_vocab = "$LABEL0";

std::string const emws_seger::label1_as_vocab = "$LABEL1";

std::string const emws_seger::unknown_as_vocab = "$OOV";

std::string const emws_seger::su_prefix = "$SU";

std::string const emws_seger::sb_prefix = "$SB";

std::tuple<char, char> const emws_seger::state_varient{'0', '1'};

