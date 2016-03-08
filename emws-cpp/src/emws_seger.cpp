//
// Created by junfeng on 3/2/16.
//

#include <iostream>
#include <algorithm>
#include <functional>
#include <random>

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
    min_alpha = config["min_alpha"].GetDouble();
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
    using namespace std;
    logger->info("start train ...");
    if (train_corpus.size() > 0) {
        // build vocabulary
        build_vocab(train_corpus);
        unsigned chunksize = 200;
        std::mt19937 g;
        g.seed(seed);
        for (epoch = 0; epoch < iter; ++epoch) {
            train_mode = true;
            // shuffle train corpus
            std::shuffle(train_corpus.begin(), train_corpus.end(), g);

            logger->info("training at epoch %v", epoch+1);
            do_train(train_corpus, chunksize, epoch);

            train_mode = false;

            // TODO eval on dev and test corpus
        }
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

void emws_seger::build_vocab(std::vector<std::vector<std::u32string> > const &sentences) {
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

std::map<std::u32string, Vocab> emws_seger::_vocab_from_new(
        std::vector<std::vector<std::u32string> > const &sentences) {

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
        // already removed '\r' at get line from file
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
    arma::arma_rng::set_seed(seed);
    syn0 = arma::randn(vocab.size(), size);
    if (pre_train) {
        // TODO load vector from pre trained vectors
    }
    syn1neg = arma::randn(vocab.size(), pred_size);
}

void emws_seger::do_train(std::vector<std::vector<std::u32string> > const &sentences, unsigned chunksize,
                          unsigned current_iter) {
    using namespace std;

    unsigned total_count = 0;
    double total_error = 0;
    unsigned current_batch_count = 0;
    double current_batch_error = 0;

    unsigned sentence_no = 0;
    double learning_rate = alpha;
    for (auto const &sentence : sentences) {
        // sentence is vector<u32string>
        sentence_no++;
        if(sentence_no % chunksize == 0) {
            total_count += current_batch_count;
            total_error += current_batch_error;
            logger->info("===> batch subgram error rate = %v, subgram_error/subgram count= %v/%v",
                         current_batch_error / current_batch_count, current_batch_error, current_batch_count);

            // update learning rate
            learning_rate = alpha * (1.0 -  (total_words * current_iter + total_count) / (total_words * iter));
            learning_rate = std::min(learning_rate, min_alpha);
            logger->info("this batch learning rate is %v", learning_rate);
            // TODO random eval
        }
        unsigned x;
        double y;
        std::tie(x, y) = train_gold_per_sentence(sentence, learning_rate);
        current_batch_count += x;
        current_batch_error += y;
    }
    return;
}

std::tuple<unsigned, double> emws_seger::train_gold_per_sentence(std::vector<std::u32string> const &sentence,
                                                                 double learning_rate) {
    using namespace std;
    unsigned count_sum = 0;
    double error_sum = 0.0;
    if (!sentence.empty()) {
        // train using the sentence
        vector<unsigned> indices;
        unsigned acc = 0;
        for (auto const &word : sentence) {
            indices.push_back(acc);
            acc += word.size();
        }
        auto char32_t_seg = str_op::join(sentence, U"");
        count_sum = static_cast<unsigned>(char32_t_seg.size());
        auto seg = str_op::full2halfwidth(char32_t_seg);
        vector<unsigned> label_list(seg.size() + 2, 1);
        label_list[label_list.size() - 1] = 0;
        label_list[label_list.size() - 2] = 0;
        for (auto i : indices)
            label_list[i] = 0;
        unsigned prev2_label = 0;
        unsigned prev_label = 0;

        for (unsigned pos = 0; pos < count_sum; ++pos) {
            // TODO continue after coding predict_single_position
            predict_single_position(seg, pos, prev2_label, prev_label, label_list);
        }
    }

    return std::make_tuple(count_sum, error_sum);
}

void emws_seger::predict_single_position(std::vector<std::u32string> &sent, unsigned pos, unsigned prev2_label,
                                         unsigned prev_label, std::vector<unsigned int> states) {
    using namespace std;
    auto future_label = states[pos + 1];
    auto future2_label = states[pos + 2];
    arma::mat feature_vec;
    arma::uvec feature_indices;
    // TODO gen_feature

    arma::uvec block;
    if (train_mode && drop_out) {
        // TODO drop out function
    }
    else if (drop_out) {
        feature_vec *= (1 - dropout_rate);
    }


    if (!block.is_empty()) {
        // logger->info("block = %v", block);
    }

    auto u = pos < sent.size() ? sent[pos] : END;
    vector<u32string> pred_words;
    for (auto const varient : state_varient) {
        u32string temp = su_prefix;
        temp += varient;
        temp += u;
        pred_words.push_back(temp);
    }

    if (vocab.count(pred_words[0]) && vocab.count(pred_words[1])) {
        // nothing to do
    }
    else {
        pred_words.clear();
        if (train_mode) {
            wstring_convert<codecvt_utf8<char32_t>, char32_t> conv;
            logger->error("Unknown candidate for %v! Should NOT happen during training!", conv.to_bytes(u));
        }
    }
    vector<u32string> pred2_words{label0_as_vocab, label1_as_vocab};
    arma::mat softmax_score;
    arma::uvec pred_indices;
    arma::mat pred_matrix;
    if (!pred_words.empty()) {
        for (auto const &pred : pred_words) {
            pred_indices << vocab[pred].index;
        }
        pred_matrix = syn1neg.rows(pred_indices);
        if (!block.is_empty())
            // TODO
            pred_matrix = block % pred_matrix;
        else if (drop_out) {
            pred_matrix = (1 - dropout_rate) * pred_matrix;
        }
        auto raw_score = arma::exp(feature_vec * pred_matrix.t());
        softmax_score = raw_score / arma::sum(raw_score);
    }
    arma::uvec pred2_indices;
    arma::mat pred2_matrix;
    arma::mat softmax2_score;

    for (auto const &pred : pred2_words) {
        pred2_indices << vocab[pred].index;
    }
    pred2_matrix = syn1neg.rows(pred2_indices);
    if (!block.is_empty()) {
        pred2_matrix = block % pred2_matrix;
    }
    else if (drop_out) {
        pred2_matrix = (1 - dropout_rate) * pred2_matrix;
    }
    auto raw2_score = arma::exp(feature_vec * pred2_matrix.t());
    softmax2_score = raw2_score / arma::sum(raw2_score);

    if (!pred_words.empty()) {
        // concate
        softmax2_score = arma::join_rows(softmax2_score, softmax_score);
        pred2_indices = arma::join_cols(pred2_indices, pred_indices);
        pred2_matrix = arma::join_cols(pred2_matrix, pred_matrix);
    }

    // TODO return
    return;
}






