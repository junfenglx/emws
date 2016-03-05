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
    static std::vector<std::u32string> split(std::u32string const &line, char32_t delim);
    static std::u32string join(std::vector<std::u32string> const &words, char32_t delim);
};


#endif //ZH_SEGER_STR_OP_H
