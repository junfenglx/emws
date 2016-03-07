//
// Created by junfeng on 3/2/16.
//

#include <iostream>
#include <algorithm>
#include <functional>

#include "emws_seger.h"

#include "utf8_io.h"
#include "str_op.h"
#include "Vocab.h"

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
    train_path = config["train_path"].GetString();

    test_raw_path = config["test_raw_path"].GetString();
    test_path = config["test_path"].GetString();
    dev_path = config["dev_path"].GetString();
    quick_test = config["quick_test"].GetString();
    dict_path = config["dict_path"].GetString();

    score_script_path = config["score_script_path"].GetString();
    pre_train = config["pre_train"].GetBool();
    uni_path = config["uni_path"].GetString();
    bi_path = config["bi_path"].GetString();
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

    logger->info("loading train, test, dev corpus ...");
    train_corpus = utf8_io::readwords(train_path);
    test_corpus = utf8_io::readlines(test_raw_path);
    dev_corpus = utf8_io::readwords(dev_path);
    quick_test_corpus = utf8_io::readlines(quick_test);

    logger->info("train_corpus num of lines: %v", train_corpus.size());
    logger->info("test_corpus num of lines: %v", test_corpus.size());

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
    if (train_corpus.size() > 0) {
        // build vocabulary
        build_vocab(train_corpus);

    }
}

std::vector<std::u32string> emws_seger::predict(std::u32string const &sentence) const {
    return std::vector<std::u32string>();
}

bool emws_seger::save(std::string const &model_path) const {
    return false;
}

el::Logger *emws_seger::logger = el::Loggers::getLogger("emws_seger");

std::u32string const emws_seger::START = U"#S#";

std::u32string const emws_seger::END = U"#E#";

std::u32string const emws_seger::label0_as_vocab = U"$LABEL0";

std::u32string const emws_seger::label1_as_vocab = U"$LABEL1";

std::u32string const emws_seger::unknown_as_vocab = U"$OOV";

std::u32string const emws_seger::su_prefix = U"$SU";

std::u32string const emws_seger::sb_prefix = U"$SB";

std::array<char32_t, 2> const emws_seger::state_varient{U'0', U'1'};

void emws_seger::build_vocab(std::vector<std::vector<std::u32string> > sentences) {
    using namespace std;
    auto new_vocab = _vocab_from_new(sentences);
    array<u32string, 3> meta_words = {label0_as_vocab, label1_as_vocab, unknown_as_vocab};
    for (auto const &meta_word : meta_words) {
        Vocab v(1);
        v.index = vocab.size();
        vocab[meta_word] = v;
        index2word.push_back(meta_word);
    }
    // TODO iter new_vocab
    for (auto const &item : new_vocab) {
        auto word = item.first;
        auto v = item.second;
        if (v.count >= min_count) {
            index2word.push_back(word);
            vocab[word] = v;
        }
    }

    // hybrid_pred
    if (hybrid_pred) {
        vector<unsigned> freq_vec;
        for (auto const &item : vocab) {
            auto word = item.first;
            auto v = item.second;
            if (word.size() == 1)
                freq_vec.push_back(v.count);
        }
        if (!freq_vec.empty()) {
            sort(freq_vec.begin(), freq_vec.end(), greater<unsigned>());
            hybrid_threshold = freq_vec[freq_vec.size() / 25];
            logger->info("frequencey threshold for hybrid prediction is: %v", hybrid_threshold);
        }
        else {
            logger->warn("freq_vec is empty, not initialize field hybrid_threshold");
        }
    }
    logger->info("total %v word types after removing those with count < %v", vocab.size(), min_count);
    logger->info("reset weights ...");
    reset_weights();
}

std::map<std::u32string, Vocab> emws_seger::_vocab_from_new(std::vector<std::vector<std::u32string> > sentences) {
    using namespace std;

    logger->info("collecting all words and their counts");

    map<u32string, Vocab> new_vocab;
    unsigned sentence_no = 0;
    total_words = 0;

    for ( auto const &sentence : train_corpus) {
        // sentence is vector<u32string>
        sentence_no++;
        if(sentence_no % 200 == 0)
            logger->info("PROGRESS: at sentence #%v, processed %v words and %v word types",
                         sentence_no, total_words, new_vocab.size());

        // remove '\r' character
        auto seq = str_op::full2halfwidth(str_op::join(sentence, U""));
        if (!seq.empty()) {
            vector<u32string> char32_t_seq = {START, START};

            // Python extern method
            char32_t_seq.insert(char32_t_seq.end(), seq.begin(), seq.end());
            // initializer list
            char32_t_seq.insert(char32_t_seq.end(), {END, END});
            // sentence has tail '\r
            total_words += (char32_t_seq.size() - 3);

            auto subgrams = char32_t_seq;
            for ( auto const varient : state_varient) {
                for (auto const u32c : char32_t_seq) {
                    u32string temp(su_prefix);
                    temp += varient;
                    temp += u32c;
                    subgrams.push_back(temp);
                }
            }
            vector<u32string> bigrams;
            for (auto iter = char32_t_seq.cbegin(); iter != char32_t_seq.cend() - 1; ++iter) {
                u32string biword(*iter);
                biword += *(iter +1);
                bigrams.push_back(biword);
            }
            for ( auto const varient : state_varient) {
                for (auto const &biword : bigrams) {
                    u32string temp(sb_prefix);
                    temp += varient;
                    temp += biword;
                    subgrams.push_back(temp);
                }
            }
            subgrams.insert(subgrams.end(), bigrams.begin(), bigrams.end());
            // end construct subgrams

            for (auto const &sub : subgrams) {
                if (new_vocab.count(sub)) {
                    // already in, update count
                    new_vocab[sub].count++;
                }
                else {
                    new_vocab[sub] = Vocab(1);
                }
            }
        }
    }
    logger->info("collected %v word types from a corpus of %v words and %v sentences",
                 new_vocab.size(), total_words, sentence_no);

    return new_vocab;
}

void emws_seger::reset_weights() {
    logger->info("resetting layer weights");
    syn0 = arma::randn(vocab.size(), size);
    if (pre_train) {
        // TODO load vector from pre trained vectors
    }
    syn1neg = arma::randn(vocab.size(), pred_size);
}
