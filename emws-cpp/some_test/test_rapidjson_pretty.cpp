//
// Created by junfeng on 3/3/16.
//

// JSON pretty formatting example
// This example can only handle UTF-8. For handling other encodings, see prettyauto example.

#include <cstdio>
#include <cstdlib>

#include "rapidjson/prettywriter.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/document.h"


int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "%s path_to_generate\n", argv[0]);
        exit(1);
    }

    using namespace rapidjson;
    using namespace std;

    Document d;
    Document::AllocatorType &a = d.GetAllocator();
    d.SetObject();
    d.AddMember("size", 50, a);
    d.AddMember("alpha", 0.1, a);
    d.AddMember("min_alpha", 0.0001, a);
    d.AddMember("min_count", 1, a);
    d.AddMember("seed", 1, a);
    d.AddMember("workers", 1, a);
    d.AddMember("iter", 10, a);
    d.AddMember("use_gold", false, a);

    d.AddMember("train_path", "null", a);
    d.AddMember("test_raw_path", "null", a);
    d.AddMember("test_path", "null", a);
    d.AddMember("dev_path", "null", a);
    d.AddMember("quick_test", "null", a);
    d.AddMember("dict_path", "null", a);
    d.AddMember("score_script_path", "null", a);

    d.AddMember("pre_train", false, a);
    d.AddMember("uni_path", "null", a);
    d.AddMember("bi_path", "null", a);

    d.AddMember("hybrid_pred", true, a);
    d.AddMember("no_action_feature", false, a);
    d.AddMember("no_bigram_feature", false, a);
    d.AddMember("no_unigram_feature", false, a);
    d.AddMember("no_binary_action_feature", true, a);
    d.AddMember("no_sb_state_feature", false, a);
    d.AddMember("no_right_action_feature", true, a);

    // Prepare writer and output stream.
    FILE* fp = fopen(argv[1], "wb");
    char writeBuffer[65536];
    FileWriteStream fos(fp, writeBuffer, sizeof(writeBuffer));
    PrettyWriter<FileWriteStream> fwriter(fos);
    d.Accept(fwriter);
    fclose(fp);

    // write to stdout again
    FileWriteStream os(stdout, writeBuffer, sizeof(writeBuffer));
    PrettyWriter<FileWriteStream> writer(os);
    d.Accept(writer);
    fprintf(stdout, "\n");
    return 0;
}