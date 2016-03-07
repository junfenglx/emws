//
// Created by junfeng on 3/5/16.
//

#include "utf8_io.h"
#include "str_op.h"

std::vector<std::u32string> utf8_io::readlines(std::string const &filename) {
    using namespace std;
    ifstream in_f = ifstream(filename, std::ios_base::binary);
    wstring_convert<codecvt_utf8<char32_t>, char32_t> conv;
    vector<u32string> sentences;
    string sentence;
    while (getline(in_f, sentence)) {
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
        // cout << sentence << endl;
        u32string u32sentence = conv.from_bytes(sentence);
        vector<u32string> words = str_op::split(u32sentence, delim);
        vec_words.push_back(words);
    }
    return vec_words;
}

bool utf8_io::writelines(std::string const &filename, std::vector<std::u32string> const &sentences, char32_t delim) {
    return false;
}

bool utf8_io::writewords(std::string const &filename, std::vector<std::vector<std::u32string>> const &vec_words,
                         char32_t word_delim, char32_t line_delim) {
    return false;
}

bool utf8_io::write(std::string const &filename, std::u32string const &content) {
    return false;
}
