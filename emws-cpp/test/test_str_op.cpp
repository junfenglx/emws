//
// Created by junfeng on 3/5/16.
//

#include "gtest/gtest.h"
#include "str_op.h"

TEST(test_str_op, test_split_sentence) {
    using namespace std;
    u32string s = U"你好，世界";
    vector<u32string> words = str_op::split(s, U'，');
    EXPECT_EQ(words[0], U"你好");
    EXPECT_EQ(words[1], U"世界");

    s = U"你好啊　C++";
    words = str_op::split(s, U'　');
    EXPECT_EQ(words[0], U"你好啊");
    EXPECT_EQ(words[1], U"C++");

    s = U"！！！如此感叹";
    words = str_op::split(s, U'！');
    EXPECT_EQ(words.size(), 4);
}

TEST(test_str_op, test_join_sentence) {
    using namespace std;
    u32string s = U"你好，世界";
    vector<u32string> words = {U"你好", U"世界"};
    EXPECT_EQ(str_op::join(words, U"，"), s);

    s = U"！！！如此感叹";
    words = {U"",U"", U"", U"如此感叹"};
    EXPECT_EQ(str_op::join(words, U"！"), s);

}

TEST(test_str_op, test_join_sentences) {
    using namespace std;
    u32string s = U"你好，世界";
    vector<u32string> words = {U"你好", U"世界"};
    vector<vector<u32string>> vec_words = {words, {U"你好啊", U"C++"}};
    vector<u32string> ret = str_op::join(vec_words, U"，");
    EXPECT_EQ(ret[0], s);
    u32string s2 = U"你好啊，C++";
    EXPECT_EQ(ret[1], s2);

    u32string content = s + U'\n' + s2;
    u32string ret_content = str_op::join(vec_words, U"，", U"\n");
    EXPECT_EQ(ret_content, content);
}