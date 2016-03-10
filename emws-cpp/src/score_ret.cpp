//
// Created by junfeng on 3/10/16.
//

#include <locale>
#include <codecvt>

#include "score_ret.h"

#include "utf8_io.h"
#include "str_op.h"

score_ret score_ret::parse_evaluation_result(std::string const &path_to_evaluation_result) {
    // parse score result
    using namespace std;
    auto lines = utf8_io::readlines(path_to_evaluation_result);
    score_ret sret;
    wstring_convert<codecvt_utf8<char32_t>, char32_t> conv;
    if (!lines.empty()) {
        auto last_line = lines.back();
        auto las_line_tokens = str_op::split(last_line, U'\t');
        if (las_line_tokens.at(0) == U"###" && las_line_tokens.size() == 14) {
            // format right
            vector<u32string> measures(las_line_tokens.end() - 6, las_line_tokens.end());
            auto recall = measures.at(0);
            auto precision = measures.at(1);
            auto f_score = measures.at(2);
            auto oov_rate = measures.at(3);
            auto oov_recall = measures.at(4);
            auto iv_recall = measures.at(5);
            sret = score_ret(conv.to_bytes(f_score), conv.to_bytes(oov_recall), conv.to_bytes(iv_recall));
        }
        else {
            // format unknowns
            logger->error("Error! Format of the EVALUATION RESULT does not match the standard!");
        }

    }
    else {
        logger->error("file %v has no content", path_to_evaluation_result);
    }
    return sret;
}

el::Logger *score_ret::logger = el::Loggers::getLogger("score_ret");