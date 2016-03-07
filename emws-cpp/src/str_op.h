//
// Created by junfeng on 3/5/16.
//

#ifndef ZH_SEGER_STR_OP_H
#define ZH_SEGER_STR_OP_H

#include <string>
#include <sstream>
#include <vector>

class str_op {
public:
    static std::vector<std::u32string> split(std::u32string const &sentence, char32_t delim);

    static std::vector<std::vector<std::u32string>> split(std::vector<std::u32string> const &sentences, char32_t delim);

    static std::u32string join(std::vector<std::u32string> const &words, char32_t delim);

    static std::vector<std::u32string> join(std::vector<std::vector<std::u32string>> const &vec_words, char32_t delim);

    static std::u32string join(
            std::vector<std::vector<std::u32string>> const &vec_words,
            char32_t word_delim, char32_t line_delim);
};


#endif //ZH_SEGER_STR_OP_H
