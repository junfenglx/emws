//
// Created by junfeng on 3/5/16.
//

#include <iostream>
#include <locale>

#include "str_op.h"


std::vector<std::u32string> str_op::split(std::u32string const &sentence, char32_t delim) {
    using namespace std;
    basic_istringstream<char32_t> iss(sentence);
    u32string word;
    vector<u32string> words;
    while (getline<char32_t>(iss, word, delim)) {
        // doesn't remove empty word
        // respect for Python split semantic
        words.push_back(word);
    }
    return words;
}


std::vector<std::u32string> str_op::split(std::u32string const &sentence) {
    using namespace std;
    basic_istringstream<char32_t> iss(sentence);
    u32string word;
    vector<u32string> words;
    while (getline<char32_t>(iss, word, U' ')) {
        // none parameter split
        // respect for Python split semantic
        // remove empty word
        if (!word.empty())
            words.push_back(word);
    }
    return words;
}

std::vector<std::vector<std::u32string>> str_op::split(std::vector<std::u32string> const &sentences, char32_t delim) {
    using namespace std;
    vector<vector<u32string>> vec_words(sentences.size());
    for (auto const &sentence : sentences) {
        vec_words.push_back(split(sentence, delim));
    }
    return vec_words;
}

std::vector<std::vector<std::u32string>> str_op::split(std::vector<std::u32string> const &sentences) {
    using namespace std;
    vector<vector<u32string>> vec_words(sentences.size());
    for (auto const &sentence : sentences) {
        vec_words.push_back(split(sentence));
    }
    return vec_words;
}

std::u32string str_op::join(std::vector<std::u32string> const &words, std::u32string const &delim) {
    using namespace std;
    basic_ostringstream<char32_t> oss;
    for (auto iter = words.cbegin(); iter != words.cend(); ++iter) {
        if (iter != words.cbegin())
            oss << delim;
        oss << *iter;
    }
    return oss.str();
}

std::vector<std::u32string> str_op::join(std::vector<std::vector<std::u32string>> const &vec_words,
                                         std::u32string const &delim) {
    using namespace std;
    vector<u32string> sentences;
    // cout << endl;
    for (auto const &words : vec_words) {
        u32string sentence = join(words, delim);
        // cout << words.size() << endl;
        // cout << sentence.size() << endl;
        sentences.push_back(sentence);
    }
    return sentences;
}

std::u32string str_op::join(std::vector<std::vector<std::u32string>> const &vec_words,
                            std::u32string const &word_delim, std::u32string const &line_delim) {
    using namespace std;
    basic_ostringstream<char32_t> oss;
    for (auto vec_iter = vec_words.cbegin(); vec_iter != vec_words.cend(); ++vec_iter) {
        if (vec_iter != vec_words.cbegin())
            oss << line_delim;

        auto &words = *vec_iter;
        for (auto word_iter = words.cbegin(); word_iter != words.cend(); ++word_iter) {
            if (word_iter != words.cbegin())
                oss << word_delim;
            oss << *word_iter;
        }
    }
    return oss.str();
}

std::u32string str_op::strip(std::u32string const &word, char32_t c) {
    using namespace std;
    bool *flags = new bool[word.size()]{false};
    bool leading = true;
    decltype(word.size()) i = 0;
    auto start = word.cbegin();
    while (start != word.cend()) {
        if (*start != c) break;
        start++;
    }
    auto end = word.cend();
    end;
    while (end != word.cbegin()) {
        auto cur = end - 1;
        if (*cur != c) break;
        end--;
    }
    u32string striped;
    if (start < end) {
        striped = word.substr(start - word.cbegin(), end - start);
    }
    return striped;
}

std::vector<std::u32string> str_op::full2halfwidth(std::u32string const &sentence) {
    using namespace std;
    vector<u32string> v;
    for (auto c : sentence) {
        if (c == U'\r' || c == U' ') {
            // cout << c << endl;
            continue;
        }
        if (c >= 65280 && c < 65375)
            c -= 65248;
        v.push_back({c});
    }
    return v;
}
