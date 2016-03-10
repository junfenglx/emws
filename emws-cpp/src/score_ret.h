//
// Created by junfeng on 3/10/16.
//

#ifndef ZH_SEGER_SCORE_RET_H
#define ZH_SEGER_SCORE_RET_H

#include <string>
#include <easylogging++.h>

class score_ret {
public:
    score_ret() = default;
    score_ret(const std::string &f_score, const std::string &oov_recall, const std::string &iv_recall) : f_score(
            f_score), oov_recall(oov_recall), iv_recall(iv_recall) { }

    static score_ret parse_evaluation_result(std::string const &path_to_evaluation_result);

public:
    std::string f_score = "--";
    std::string oov_recall = "--";
    std::string iv_recall = "--";

    static el::Logger *logger;

};


#endif //ZH_SEGER_SCORE_RET_H
