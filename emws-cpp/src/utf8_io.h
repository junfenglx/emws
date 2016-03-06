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

    static bool writelines(std::string const &filename, std::vector<std::u32string> const &sentences, char32_t delim);

    static bool writewords(std::string const &filename, std::vector<std::vector<std::u32string>> const &vec_words,
                           char32_t word_delim, char32_t line_delim);

    static bool write(std::string const &filename, std::u32string const &content);
};


#endif //ZH_SEGER_UTF8_IO_H
