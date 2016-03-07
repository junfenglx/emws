//
// Created by junfeng on 3/5/16.
//

#include <string>
#include <iostream>
#include <vector>
#include <locale>
#include <codecvt>


#include "str_op.h"

int main() {
    using namespace std;
    wstring_convert<codecvt_utf8<char32_t>, char32_t> conv;
    string line = "你好啊世界";
    u32string u32line = conv.from_bytes(line);
    char32_t delim = u'啊';
    vector<u32string> words = str_op::split(u32line, delim);
    for (auto const &word : words) {
        cout << conv.to_bytes(word) << endl;
    }
}