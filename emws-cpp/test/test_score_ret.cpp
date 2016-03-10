//
// Created by junfeng on 3/10/16.
//

#include <string>

#include "easylogging++.h"

INITIALIZE_EASYLOGGINGPP

#include "gtest/gtest.h"

#include "score_ret.h"


TEST(test_score_ret, test_score_ret_parse) {
    using namespace std;
    string path_to_eval = "./conf/score_output.txt";
    score_ret sret = score_ret::parse_evaluation_result(path_to_eval);
    EXPECT_EQ(sret.f_score, "0.949");
    EXPECT_EQ(sret.oov_recall, "0.754");
    EXPECT_EQ(sret.iv_recall, "0.955");
}