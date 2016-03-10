//
// Created by junfeng on 3/5/16.
//

#include <iomanip>
#include <ctime>

#include "utf8_io.h"
#include "str_op.h"


static inline void handle_win_file(std::string &sentence) {
    if (!sentence.empty() && sentence.back() == '\r') {
        sentence.pop_back();
    }
}

std::vector<std::u32string> utf8_io::readlines(std::string const &filename) {
    using namespace std;
    ifstream in_f = ifstream(filename, std::ios_base::binary);
    wstring_convert<codecvt_utf8<char32_t>, char32_t> conv;
    vector<u32string> sentences;
    string sentence;
    while (getline(in_f, sentence)) {
        handle_win_file(sentence);
        // cout << sentence << endl;
        sentences.push_back(conv.from_bytes(sentence));
    }
    return sentences;
}

std::vector<std::vector<std::u32string>> utf8_io::readwords(std::string const &filename, char32_t delim) {
    using namespace std;
    ifstream in_f = ifstream(filename, std::ios_base::binary);
    wstring_convert<codecvt_utf8<char32_t>, char32_t> conv;
    vector<vector<u32string>> vec_words;
    string sentence;
    while (getline(in_f, sentence)) {
        handle_win_file(sentence);
        // cout << sentence << endl;
        u32string u32sentence = conv.from_bytes(sentence);
        vector<u32string> words = str_op::split(u32sentence, delim);
        vec_words.push_back(words);
    }
    return vec_words;
}


std::vector<std::vector<std::u32string>> utf8_io::readwords(std::string const &filename) {
    using namespace std;
    ifstream in_f = ifstream(filename, std::ios_base::binary);
    wstring_convert<codecvt_utf8<char32_t>, char32_t> conv;
    vector<vector<u32string>> vec_words;
    string sentence;
    while (getline(in_f, sentence)) {
        handle_win_file(sentence);
        // cout << sentence << endl;
        u32string u32sentence = conv.from_bytes(sentence);
        vector<u32string> words = str_op::split(u32sentence);
        vec_words.push_back(words);
    }
    return vec_words;
}


bool utf8_io::writelines(std::string const &filename, std::vector<std::u32string> const &sentences, char32_t delim) {
    using namespace std;
    ofstream out_f = ofstream(filename, std::ios_base::binary);
    if (!out_f || out_f.bad())
        return false;
    wstring_convert<codecvt_utf8<char32_t>, char32_t> conv;
    for (auto const &sentence : sentences) {
        out_f << conv.to_bytes(sentence) << conv.to_bytes(delim);
    }
    out_f.close();
    return true;
}

bool utf8_io::writewords(std::string const &filename, std::vector<std::vector<std::u32string>> const &vec_words,
                         char32_t word_delim, char32_t line_delim) {
    using namespace std;
    ofstream out_f = ofstream(filename, std::ios_base::binary);
    if (!out_f || out_f.bad())
        return false;
    wstring_convert<codecvt_utf8<char32_t>, char32_t> conv;

    for (auto const &words : vec_words) {
        auto sentence = str_op::join(words, u32string{word_delim});
        out_f << conv.to_bytes(sentence) << conv.to_bytes(line_delim);
    }
    out_f.close();
    return true;
}

bool utf8_io::write(std::string const &filename, std::u32string const &content) {
    using namespace std;
    ofstream out_f = ofstream(filename, std::ios_base::binary);
    if (!out_f || out_f.bad())
        return false;
    wstring_convert<codecvt_utf8<char32_t>, char32_t> conv;
    string b = conv.to_bytes(content);
    out_f.write(b.c_str(), b.size());
    out_f.close();
    return false;
}

const std::string utf8_io::gen_model_path(std::string const &model_name, const std::string &base_dir) {
    using namespace std;
    time_t t = time(nullptr);
    tm t_m = *localtime(&t);
    auto const now_str = put_time(&t_m, "%F_%H-%M");
    std::ostringstream ate(std::ios_base::ate);
    ate << base_dir << '/' << model_name << '_' << now_str << ".model";
    return ate.str();
}

const std::string utf8_io::gen_temp_seged_path(std::string const &model_name, const std::string &base_dir) {
    using namespace std;
    time_t t = time(nullptr);
    tm t_m = *localtime(&t);
    auto const now_str = put_time(&t_m, "%F_%H-%M-%S");
    std::ostringstream ate(std::ios_base::ate);
    ate << base_dir << '/' << model_name << '_' << now_str << "seged.tmp";
    return ate.str();
}
