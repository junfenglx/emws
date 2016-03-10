//
// Created by junfeng on 3/5/16.
//

#include <string>
#include <iostream>
#include <vector>
#include <locale>
#include <codecvt>

#include "gtest/gtest.h"

#include "utf8_io.h"

TEST(test_seger_utils, test_readlines) {
    using namespace std;
    wstring_convert<codecvt_utf8<char32_t>, char32_t> conv;
    vector<u32string> sentences = utf8_io::readlines("./conf/emws_config.json");
    for (auto const &sentence : sentences) {
        cout << conv.to_bytes(sentence) << endl;
    }
}

TEST(test_seger_utils, test_writewords) {
    using namespace std;
    wstring_convert<codecvt_utf8<char32_t>, char32_t> conv;
    auto words_vec = utf8_io::readwords("./conf/emws_config.json", U' ');

    bool ret= utf8_io::writewords("./conf/emws_config.dup.json", words_vec, U' ', U'\n');
    EXPECT_EQ(ret, true);
}