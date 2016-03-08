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

    static std::vector<std::u32string> split(std::u32string const &sentence);

    static std::vector<std::vector<std::u32string>> split(std::vector<std::u32string> const &sentences, char32_t delim);

    static std::vector<std::vector<std::u32string>> split(const std::vector<std::u32string> &sentences);

    static std::u32string join(std::vector<std::u32string> const &words, std::u32string const &delim);

    static std::vector<std::u32string> join(std::vector<std::vector<std::u32string>> const &vec_words,
                                            std::u32string const &delim);

    static std::u32string join(
            std::vector<std::vector<std::u32string>> const &vec_words,
            std::u32string const &word_delim, std::u32string const &line_delim);

    static std::u32string strip(std::u32string const &word, char32_t c);

    static std::vector<std::u32string> full2halfwidth(std::u32string const &sentence);
};


#endif //ZH_SEGER_STR_OP_H
