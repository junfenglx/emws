//
// Created by junfeng on 3/5/16.
//

#include <string>
#include <iostream>
#include <vector>
#include <locale>
#include <codecvt>

#include "utf8_io.h"

template <class ChT>
std::vector<std::basic_string<ChT>> readlines(std::string const &filename, ChT line_delim) {
    using namespace std;
    basic_ifstream<ChT> in_f = basic_ifstream<ChT>(filename, std::ios_base::binary);
    vector<basic_string<ChT>> sentences;
    basic_string<ChT> sentence;
    locale l1 = in_f.getloc();
    // l1 = locale();
    cout << "default loc name is: " << l1.name() << endl;
    locale loc(l1, new codecvt_utf8<ChT>);
    cout << "the loc name is: " << loc.name() << endl;
    in_f.imbue(loc);
    for (ChT c; in_f.get(c);) {
        // cout << sentence << endl;
        if (c == line_delim) {
            sentences.push_back(sentence);
            sentence.clear();
        }
        else {
            sentence += c;
        }
    }
    return sentences;
}

template <class ChT>
std::vector<std::basic_string<ChT>> readlines(std::string const &filename) {
    using namespace std;
    basic_ifstream<ChT> in_f = basic_ifstream<ChT>(filename, std::ios_base::binary);
    vector<basic_string<ChT>> sentences;
    basic_string<ChT> sentence;
    locale l1 = in_f.getloc();
    l1 = locale("en_US.UTF8");
    cout << "default loc name is: " << l1.name() << endl;
    // auto cvt = new std::codecvt_byname<ChT, char, std::mbstate_t>("");
    auto cvt = new codecvt_utf8<ChT>();
    locale loc(l1, cvt);
    cout << "the loc name is: " << loc.name() << endl;
    in_f.imbue(loc);
    while (getline<ChT>(in_f, sentence)) {
        sentences.push_back(sentence);
    }
    return sentences;
}

int main() {
    using namespace std;
    using ChT = char32_t;
    cout << "sizeof (wchar_t): " << sizeof (wchar_t) << endl <<
            "sizeof (char32_t): " << sizeof (char32_t) << endl;
    wchar_t wc = L'你';
    char32_t c32 = U'你';
    cout << "wc == c32 " << boolalpha << (wc == c32) << endl;
    cout << hex << showbase << wc << ' ' << c32 << endl;
    cout << dec << showbase << wc << ' ' << c32 << endl;

    wstring_convert<codecvt_utf8<ChT>, ChT> conv;
    // std::bad_cast std::getline bug
    auto sentences = readlines<ChT>("./conf/emws_config.json");
    for (auto const &sentence : sentences) {
        cout << conv.to_bytes(sentence) << endl;
    }
}