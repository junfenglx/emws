//
// Created by junfeng on 3/2/16.
//

#include <iostream>
#include <algorithm>
#include <functional>
#include <random>
#include <set>
#include <cstdlib>
#include <cmath>

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

    // test passed even set to true
    drop_out = false;

    binary_pred = false;

    logger->info("loading train, test, dev corpus ...");
    train_corpus = utf8_io::readwords(train_path);
    test_corpus = utf8_io::readlines(test_raw_path);
    dev_corpus = utf8_io::readlines(dev_path);
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
        // TODO implements pre train(most low priority)
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
        // when debug, decrease chunksize
        unsigned chunksize = 200;
        std::mt19937 g;
        g.seed(seed);
        for (epoch = 0; epoch < iter; ++epoch) {
            train_mode = true;
            // shuffle train corpus
            std::shuffle(train_corpus.begin(), train_corpus.end(), g);

            logger->info("training at epoch %v", epoch + 1);
            do_train(train_corpus, chunksize, epoch);

            train_mode = false;
            // eval on dev and test corpus
            logger->info("===Eval on dev corpus at epoch %v", epoch + 1);
            score_ret sret = eval(dev_corpus, dev_path);
            logger->info("F-score = %v/OOV-Recall = %v/IV-recall = %v",
                         sret.f_score, sret.oov_recall, sret.iv_recall);

            logger->info("===Eval on test corpus at epoch %v", epoch + 1);
            score_ret test_sret = eval(test_corpus, test_path);
            logger->info("F-score = %v/OOV-Recall = %v/IV-recall = %v\n\n",
                         test_sret.f_score, test_sret.oov_recall, test_sret.iv_recall);

        }
    }
}

std::vector<std::u32string> emws_seger::predict(std::u32string const &sentence) const {
    return predict_sentence_greedy(sentence);
}

bool emws_seger::save(std::string const &model_path) const {
    // TODO serialization(high priority)
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

std::array<std::u32string, 2> const emws_seger::state_varient{U"0", U"1"};

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
    // iter new_vocab
    for (auto const &item : new_vocab) {
        auto word = item.first;
        auto v = item.second;
        v.index = vocab.size();
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
    syn0 = (arma::randu(vocab.size(), size) - 0.5) / size;
    if (pre_train) {
        // TODO load vector from pre trained vectors(most low priority)
    }
    syn1neg = arma::mat(vocab.size(), pred_size, arma::fill::zeros);
}

void emws_seger::do_train(std::vector<std::vector<std::u32string> > const &sentences, unsigned chunksize,
                          unsigned current_iter) {
    using namespace std;

    unsigned total_count = 0;
    double total_error = 0;
    unsigned current_batch_count = 0;
    double current_batch_error = 0;

    unsigned sentence_no = 0;
    auto update_lr = [this](unsigned ci, unsigned tc) -> double {
        return alpha * (1.0 -  static_cast<double>(total_words * ci + tc) / (total_words * iter));
    };
    // double learning_rate = alpha;
    double learning_rate = update_lr(current_iter, total_count);

    // batch_num
    unsigned batch_no = 0;
    unsigned total_batch = static_cast<unsigned>(sentences.size()) / chunksize;
    for (auto const &sentence : sentences) {
        // sentence is vector<u32string>
        sentence_no++;
        if(sentence_no % chunksize == 0) {
            batch_no = sentence_no / chunksize;
            total_count += current_batch_count;
            total_error += current_batch_error;
            logger->info("Epoch: %v, at batch_no %v / total_batch %v", current_iter + 1, batch_no, total_batch);
            logger->info("current batch learning rate is %v", learning_rate);
            logger->info("===> batch subgram error rate = %v, subgram_error/subgram count= %v/%v\n",
                         current_batch_error / current_batch_count,
                         current_batch_error,
                         current_batch_count);

            // update learning rate
            // learning_rate = alpha * (1.0 -  static_cast<double>(total_count) / (total_words));
            learning_rate = update_lr(current_iter, total_count);
            learning_rate = std::max(learning_rate, min_alpha);
            current_batch_count = 0;
            current_batch_error = 0;
            // TODO random eval(low priority)
        }
        unsigned x;
        double y;
        std::tie(x, y) = train_gold_per_sentence(sentence, learning_rate);
        current_batch_count += x;
        current_batch_error += y;
    }
    // last batch
    logger->info("current batch learning rate is %v", learning_rate);
    logger->info("===> batch subgram error rate = %v, subgram_error/subgram count= %v/%v\n",
                    current_batch_error / current_batch_count,
                    current_batch_error,
                    current_batch_count);
    total_count += current_batch_count;
    total_error += current_batch_error;

    logger->info("at epoch %v, processed %v words, total_words %v\n", current_iter + 1,
            total_count, total_words);
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
        vector<unsigned> label_vec(seg.size() + 2, 1);
        label_vec[label_vec.size() - 1] = 0;
        label_vec[label_vec.size() - 2] = 0;
        for (auto i : indices)
            label_vec[i] = 0;
        unsigned prev2_label = 0;
        unsigned prev_label = 0;

        for (unsigned pos = 0; pos < count_sum; ++pos) {
            arma::vec softmax_score;
            arma::uvec feature_indices;
            arma::uvec pred_indices;
            arma::rowvec feature_vec;
            arma::mat pred_matrix;
            std::tie(softmax_score, feature_indices, pred_indices, feature_vec, pred_matrix) =
                    predict_single_position(seg, pos, prev2_label, prev_label, label_vec);
            auto true_label = label_vec[pos];
            arma::vec gold_score;
            if (softmax_score.n_elem == 4) {
                assert(train_mode);
                if (true_label == 0)
                    gold_score = {1.0, 0.0, 1.0, 0.0};
                else if (true_label == 1)
                    gold_score = {0.0, 1.0, 0.0, 1.0};
                else
                    logger->error("Error! true label should either 1 or 0, but now it is: %v", true_label);
            }
            else {
                logger->error("The output of predict_single_position"
                                      " should have either 2 or 4 scores, but now it has %v", softmax_score.n_elem);
                assert(false);
            }
            arma::vec error_array = gold_score - softmax_score;
            error_sum += arma::sum(arma::abs(error_array)) / error_array.n_elem;
            if (std::isnan(error_sum))
                logger->info("error_array is %v", error_array.t());
            arma::vec gb = error_array * learning_rate;
            arma::mat neu1e = gb.t() * pred_matrix.cols(0, non_fixed_param - 1);
            if (l2_rate > 0) {
                syn1neg.rows(pred_indices) *= (1 - learning_rate * l2_rate);
                syn0.rows(feature_indices) *= (1 - learning_rate * l2_rate);
            }

            syn1neg.rows(pred_indices) += (gb * feature_vec);
            neu1e.resize(feature_indices.n_elem, neu1e.n_elem / feature_indices.n_elem);
            syn0.rows(feature_indices) += neu1e;
            softmax_score = softmax_score.subvec(softmax_score.n_elem - 2, softmax_score.n_elem - 1);
            unsigned label;
            if (softmax_score(1) > 0.5)
                label = 1;
            else
                label = 0;
            prev2_label = prev_label;
            prev_label = label;
            if (use_gold)
                prev_label = true_label;
        }
    }

    return std::make_tuple(count_sum, error_sum);
}

std::tuple<arma::vec, arma::uvec, arma::uvec, arma::rowvec, arma::mat>
emws_seger::predict_single_position(std::vector<std::u32string> &sent, unsigned pos, unsigned prev2_label,
                                    unsigned prev_label, std::vector<unsigned int> states) const {
    using namespace std;
    auto future_label = states[pos + 1];
    auto future2_label = states[pos + 2];
    arma::rowvec feature_vec;
    arma::uvec feature_indices;
    std::tie(feature_vec, feature_indices) =
            gen_feature(sent, pos, prev2_label, prev_label, future_label, future2_label);

    arma::urowvec block;
    if (train_mode && drop_out) {
        arma::Col<unsigned> to_block = arma::linspace<arma::Col<unsigned> >(0, non_fixed_param-1, non_fixed_param);
        to_block = arma::shuffle(to_block);
        to_block = to_block.subvec(0, dropout_size - 1);
        set<unsigned> in_block;
        for (unsigned i = 0; i < to_block.n_elem; ++i)
            in_block.insert(to_block(i));
        block = arma::urowvec(pred_size, arma::fill::ones);
        for (unsigned i =0; i < pred_size; ++i) {
            if (in_block.count(i))
                block(i) = 0;
        }
        feature_vec = feature_vec % block;
        // no runtime error when dropout set to true
    }
    else if (drop_out) {
        feature_vec *= (1 - dropout_rate);
    }

    arma::umat block_mat;
    if (!block.is_empty()) {
        // to verbose
        // logger->info("block = %v", block);
        block_mat = arma::join_cols(block, block);
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
    arma::vec softmax_score;
    arma::uvec pred_indices(pred_words.size());
    arma::mat pred_matrix;
    if (!pred_words.empty()) {
        unsigned i = 0;
        for (auto const &pred : pred_words) {
            pred_indices(i) = vocab.at(pred).index;
            i++;
        }
        pred_matrix = syn1neg.rows(pred_indices);
        if (!block.is_empty())
            // already debug
            pred_matrix = block_mat % pred_matrix;
        else if (drop_out) {
            pred_matrix = (1 - dropout_rate) * pred_matrix;
        }
        // feature_vec is 1*600 matrix, pred_matrix is 2*600 matrix
        // raw_score is a column vector, size is 2
        arma::vec raw_score = arma::exp(feature_vec * pred_matrix.t()).t();
        softmax_score = raw_score / arma::sum(raw_score);
    }
    arma::uvec pred2_indices(pred2_words.size());
    arma::mat pred2_matrix;
    arma::vec softmax2_score;

    unsigned i = 0;
    for (auto const &pred : pred2_words) {
        pred2_indices(i) = vocab.at(pred).index;
        i++;
    }
    pred2_matrix = syn1neg.rows(pred2_indices);
    if (!block.is_empty()) {
        // shape need same even element wise multiplication
        // Aramdillo doesn't broadcast elements
        pred2_matrix = block_mat % pred2_matrix;
    }
    else if (drop_out) {
        pred2_matrix = (1 - dropout_rate) * pred2_matrix;
    }
    arma::vec raw2_score = arma::exp(feature_vec * pred2_matrix.t()).t();
    softmax2_score = raw2_score / arma::sum(raw2_score);

    if (!pred_words.empty()) {
        // concate
        softmax2_score = arma::join_cols(softmax2_score, softmax_score);
        pred2_indices = arma::join_cols(pred2_indices, pred_indices);
        pred2_matrix = arma::join_cols(pred2_matrix, pred_matrix);
    }
    // cout << softmax2_score << endl;
    return std::make_tuple(softmax2_score, feature_indices, pred2_indices, feature_vec, pred2_matrix);
}

std::tuple<arma::rowvec, arma::uvec>
emws_seger::gen_feature(std::vector<std::u32string> &sent, unsigned pos,
            unsigned prev2_label, unsigned prev_label,
            unsigned future_label, unsigned future2_label) const {

    using namespace std;

    auto uni_bi_grams = gen_unigram_bigram(sent, pos);
    vector<u32string> ngram_feature_vec(uni_bi_grams.begin(), uni_bi_grams.end());
    if (no_bigram_feature) {
        auto last = ngram_feature_vec.end();
        ngram_feature_vec.erase(last - 4, last);
    }

    if (no_unigram_feature) {
        auto first = ngram_feature_vec.begin();
        ngram_feature_vec.erase(first, first + 5);
    }
    auto u_2 = uni_bi_grams[2];
    auto u_1 = uni_bi_grams[1];
    vector<u32string> state_feature_vec;
    state_feature_vec.push_back(
            su_prefix + state_varient[prev2_label] + u_2
    );
    state_feature_vec.push_back(
            su_prefix + state_varient[prev_label] + u_1
    );

    auto u1 = uni_bi_grams[3];
    auto u2 = uni_bi_grams[4];
    vector<u32string> right_state_feature_vec;
    right_state_feature_vec.push_back(
            su_prefix + state_varient[future_label] + u1
    );
    right_state_feature_vec.push_back(
            su_prefix + state_varient[future2_label] + u2
    );
    auto b_1 = uni_bi_grams[5];
    auto b1 = uni_bi_grams[7];
    if (!no_sb_state_feature) {
        state_feature_vec.push_back(
                sb_prefix + state_varient[prev_label] + b_1
        );
        right_state_feature_vec.push_back(
                sb_prefix + state_varient[future_label] + b1
        );
    }
    vector<u32string> feat_vec;
    if (no_action_feature) {
        feat_vec = ngram_feature_vec;
    }
    else {
        feat_vec = ngram_feature_vec;
        feat_vec.insert(feat_vec.end(), state_feature_vec.begin(), state_feature_vec.end());
    }
    if (!no_right_action_feature) {
        feat_vec.insert(feat_vec.end(), right_state_feature_vec.begin(), right_state_feature_vec.end());
    }

    auto feat_indices = words2indices(feat_vec);
    auto feature_mat = syn0.rows(feat_indices);
    auto feature_vec = arma::vectorise(feature_mat, 1);
    // feat_indices size is 12x1, feature_vec size is 1x600
    // logger->info("feat_indices size is %v, feature_vec size is %v", arma::size(feat_indices), arma::size(feature_vec));
    return std::make_tuple(feature_vec, feat_indices);
}

std::array<std::u32string, 9> emws_seger::gen_unigram_bigram(std::vector<std::u32string> &sent, unsigned pos) const {
    using namespace std;
    auto n = sent.size();
    auto u = pos < n ? sent[pos] : END;
    auto u_1 = pos > 0 ? sent[pos - 1] : START;
    auto u_2 = pos > 1 ? sent[pos - 2] : START;
    auto u1 = pos + 1 < n ? sent[pos + 1] : END;
    // C++ unsigned < signed
    auto u2 = pos + 2 < n ? sent[pos + 2] : END;
    auto b_1 = u_1 + u;
    auto b_2 = u_2 + u_1;
    auto b1 = u + u1;
    auto b2 = u1 + u2;

    return std::array<std::u32string, 9>{u, u_1, u_2, u1, u2, b_1, b_2, b1, b2};
}

arma::uvec emws_seger::words2indices(std::vector<std::u32string> const &feat_vec) const {
    using namespace std;
    arma::uvec feat_indices(feat_vec.size());
    for (unsigned i = 0; i < feat_vec.size(); ++i) {
        auto &word = feat_vec[i];
        unsigned long index;
        if (vocab.count(word))
            index = vocab.at(word).index;
        else
            index = vocab.at(unknown_as_vocab).index;
        feat_indices(i) = index;
    }
    return feat_indices;
}

score_ret emws_seger::eval(std::vector<std::u32string> const &sentences, std::string const &gold_path) const {
    using namespace std;
    auto seged_sentences = base_seger::predict(sentences);

    string temp_dir = "../working_data";
    auto tmp_path = utf8_io::gen_temp_seged_path("emws", temp_dir);
    auto tmp_eval_path = tmp_path + ".eval";

    bool flag = utf8_io::writewords(tmp_path, seged_sentences, U' ', U'\n');
    if (!flag) {
        logger->error("write words to %v error", tmp_path);
        return score_ret();
    }

    logger->info("eval with perl socre");

    ostringstream oss;
    oss << "perl " << score_script_path << ' ';
    oss << dict_path << ' ' << gold_path << ' ';
    oss << tmp_path << " > " << tmp_eval_path;
    string cmd = oss.str();
    std::system(cmd.c_str());
    score_ret sret = score_ret::parse_evaluation_result(tmp_eval_path);
    return sret;
}

std::vector<std::u32string> emws_seger::predict_sentence_greedy(std::u32string const &sentence) const {
    using namespace std;
    // implements predict_sentence_greedy
    vector<u32string> tokens;
    // remove blank space
    auto sent = str_op::join(str_op::split(sentence), U"");

    if (!sent.empty()) {
        vector<u32string> char32_t_seq = str_op::full2halfwidth(sent);
        vector<unsigned int> states(char32_t_seq.size() + 2, 1);
        states.at(states.size() - 1) = 0;
        states.at(states.size() - 2) = 0;

        auto do_greedy_predict = [this, &states, &char32_t_seq]() {
            unsigned prev2_label = 0;
            unsigned prev_label = 0;
            unsigned p = 0;
            for (auto const &c : char32_t_seq) {
                unsigned target;
                if (p == 0)
                    target = 0;
                else {
                    arma::vec softmax_score;
                    arma::uvec feature_indices;
                    arma::uvec pred_indices;
                    arma::rowvec feature_vec;
                    arma::mat pred_matrix;
                    std::tie(softmax_score, feature_indices, pred_indices, feature_vec, pred_matrix) =
                            predict_single_position(char32_t_seq, p, prev2_label, prev_label, states);
                    // only use softmax_score
                    if ( binary_pred)
                        softmax_score = softmax_score.subvec(0, 1);
                    else if (hybrid_pred) {
                        if (vocab.count(c) && vocab.at(c).count > hybrid_threshold)
                            softmax_score = softmax_score.subvec(softmax_score.n_elem - 2, softmax_score.n_elem - 1);
                        else {
                            arma::vec x = softmax_score.head(2);
                            arma::vec y = softmax_score.tail(2);
                            softmax_score << (x(0) + y(0)) / 2.0 << (x(1) + y(1)) / 2.0;
                        }
                    }
                    else
                        softmax_score = softmax_score.subvec(softmax_score.n_elem - 2, softmax_score.n_elem - 1);

                    // transform score to binary target
                    if (softmax_score(1) > 0.5)
                        target = 1;
                    else
                        target = 0;
                }
                // update the label in the current iter
                states[p] = target;
                prev2_label = prev_label;
                prev_label = target;
                ++p;
            }
        };
        if (no_right_action_feature)
            do_greedy_predict();
        else {
            for (unsigned i = 0; i < iter; ++i)
                do_greedy_predict();
        }
        unsigned pos = 0;
        for (auto const c : sent) {
            unsigned label = states[pos];
            if (label == 0)
                tokens.push_back({c});
            else {
                if (!tokens.empty())
                    tokens.back() += c;
                else {
                    // never reach here
                    tokens.push_back({c});
                    logger->error("should not happen! the action of the first char in the sent is \"APPEND!\"");
                }
            }
            // fuck
            // I need enumerate function in C++
            // really I feel like fuck a dog
            ++pos;
        }
    }
    return tokens;
}









