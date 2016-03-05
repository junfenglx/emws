//
// Created by junfeng on 3/5/16.
//

#include "str_op.h"


std::vector<std::u32string> str_op::split(std::u32string const &line, char32_t delim) {
    using namespace std;
    basic_istringstream<char32_t> iss(line);
    u32string word;
    vector<u32string> words;
    while (getline<char32_t>(iss, word, delim)) {
        words.push_back(word);
    }
    return words;
}

std::u32string str_op::join(std::vector<std::u32string> const &words, char32_t delim) {
    using namespace std;
    return __cxx11::basic_string < char32_t, char_traits < _CharT >, allocator < _CharT >> ();
}
