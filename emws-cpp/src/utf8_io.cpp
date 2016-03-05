//
// Created by junfeng on 3/5/16.
//

#include "utf8_io.h"

std::vector<std::u32string> utf8_io::readlines(std::string const &filename) {
    using namespace std;
    ifstream in_f = ifstream(filename, std::ios_base::binary);
    wstring_convert<codecvt_utf8<char32_t>, char32_t> conv;
    vector<basic_string<char32_t>> sentences;
    string sentence;
    while (getline(in_f, sentence)) {
        // cout << sentence << endl;
        sentences.push_back(conv.from_bytes(sentence));
    }
    return sentences;
}

std::vector<std::vector<std::u32string>> utf8_io::readwords(std::string const &filename, char32_t delim) {
    using namespace std;
    return std::vector<vector < std::__cxx11::u32string, allocator < std::__cxx11::u32string>>>();
}
