//
// Created by junfeng on 3/2/16.
//

#include <iostream>
#include <sstream>
#include <string>
#include <cstdio>
#include <ctime>
#include <iomanip>

#include <boost/program_options.hpp>
#include <rapidjson/prettywriter.h>

#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/stringbuffer.h"

#define ELPP_DEFAULT_LOG_FILE "logs/emws-predict-%datetime.log"

#include "easylogging++.h"

INITIALIZE_EASYLOGGINGPP


#include "base_seger.h"
#include "utf8_io.h"

int main(int argc, const char *argv[]) {
    using namespace std;
    using namespace rapidjson;
    using namespace boost::program_options;

    START_EASYLOGGINGPP(argc, argv);
    el::Logger* mainLogger = el::Loggers::getLogger("main");
    // mainLogger->debug("Hello, main logger");
    // CLOG(ERROR, "main") << "CLOG using main logger";

    try
    {
        options_description desc{"Options"};
        desc.add_options()
                ("help,h", "show help message")
                ("model,m", value<string>(), "model file")
                ("file,f", value<string>(), "file to be segmented")
                ("output,o", value<string>(), "path to output segmented file");

        variables_map vm;
        store(parse_command_line(argc, argv, desc), vm);
        notify(vm);

        if (vm.count("help")) {
            cout << desc << endl;
            exit(0);
        }
        if (!vm.count("model") || vm.count("file")) {
            cerr << "Must specify model file path and file path to be segmented ." << endl;
            cout << desc << endl;
            exit(1);
        }

        string const model_path = vm["model"].as<string>();
        mainLogger->info("use model file: %v", model_path);
        string const document_path = vm["file"].as<string>();
        mainLogger->info("file to be segmented: %v", document_path);
        string output_path;
        if (vm.count("output")) {
            output_path = vm["output"].as<string>();
        }
        else
            output_path = document_path + ".seged";
        mainLogger->info("segmented output path: %v", output_path);
        mainLogger->info("load model ...");
        string const model_name = "emws";
        base_seger *seger = base_seger::load(model_path, model_name);

        mainLogger->info("reading sentences from document ...");
        vector<u32string> sentences = utf8_io::readlines(document_path);

        mainLogger->info("predict ...");
        auto seged_sentences = seger->predict(sentences);

        mainLogger->info("save segmented document ...");
        bool flag = utf8_io::writewords(output_path, seged_sentences, U' ', U'\n');
        if (!flag) {
            mainLogger->error("write words to %v error", output_path);
        }

        delete seger;
    }
    catch (const error &ex)
    {
        std::cerr << ex.what() << '\n';
    }

}