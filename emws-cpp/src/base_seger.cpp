//
// Created by junfeng on 3/2/16.
//

#include <vector>

#include "base_seger.h"
#include "emws_seger.h"

base_seger * base_seger::create_seger(rapidjson::Document const &config, std::string const &seger_name) {
    base_seger *seger;
    if (seger_name == "emws") {
        seger =  new emws_seger(config);
    }
    else {
        std::string msg{"unknown seger name: "};
        msg += seger_name;
        throw msg;
    }
    return seger;
}

base_seger *base_seger::load(std::string const &model_path, std::string const &seger_name) {
    base_seger *seger;
    if (seger_name == "emws") {
        seger =  new emws_seger(model_path);
    }
    else {
        std::string msg{"unknown seger name: "};
        msg += seger_name;
        throw msg;
    }
    return seger;
}

std::vector<std::vector<std::u32string>> base_seger::predict(std::vector<std::u32string> const &sentences) {
    using namespace std;

    auto segmented_sentences = vector<vector<u32string>>();
    for (auto const &sentence : sentences) {
        vector<u32string> segmented_sentence = predict(sentence);
        segmented_sentences.push_back(segmented_sentence);
    }
    return segmented_sentences;
}
