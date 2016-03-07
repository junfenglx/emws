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