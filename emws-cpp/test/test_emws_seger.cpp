//
// Created by junfeng on 3/12/16.
//

#include <iostream>
#include <sstream>
#include <string>
#include <cstdio>
#include <ctime>
#include <iomanip>

#include "rapidjson/prettywriter.h"
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/stringbuffer.h"

#define ELPP_DEFAULT_LOG_FILE "logs/emws-unittest-%datetime.log"

#include "easylogging++.h"

INITIALIZE_EASYLOGGINGPP

#include "gtest/gtest.h"

#include "base_seger.h"
#include "utf8_io.h"

TEST(test_emws_seger, test_emws_seger_test_save) {
    el::Logger *testLogger = el::Loggers::getLogger("emws_seger_unittest");

    using namespace std;
    using namespace rapidjson;

    const string filename = "./conf/emws_config.json";
    FILE* fp = fopen(filename.data(), "rb");
    char readBuffer[65536];
    FileReadStream is(fp, readBuffer, sizeof(readBuffer));
    Document config;
    config.ParseStream(is);
    fclose(fp);

    StringBuffer buffer;
    PrettyWriter<StringBuffer> writer(buffer);
    config.Accept(writer);

    testLogger->info("=====parameters====\n%v", buffer.GetString());

    testLogger->info("initial model ...");
    string const model_name = "emws";
    base_seger *seger = base_seger::create_seger(config, model_name);

    string const model_path = utf8_io::gen_model_path(model_name, "./model");
    testLogger->info("save model to %v\n\n", model_path);
    // save model
    bool ret = seger->save(model_path);
    EXPECT_EQ(ret, true);

    // load model
    testLogger->info("load model ...");
    base_seger *loaded_seger = base_seger::load(model_path, model_name);
    testLogger->info("train ...");
    loaded_seger->train();
    testLogger->info("train done");
}