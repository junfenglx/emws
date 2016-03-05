//
// Created by junfeng on 3/5/16.
//

#ifndef ZH_SEGER_UTF8_IO_H
#define ZH_SEGER_UTF8_IO_H

#include <string>
#include <vector>
#include <fstream>
#include <locale>
#include <codecvt>
#include <iostream>


class utf8_io {
public:
    static std::vector<std::u32string> readlines(std::string const &filename);

    static std::vector<std::vector<std::u32string>> readwords(std::string const &filename, char32_t delim);
};


#endif //ZH_SEGER_UTF8_IO_H
