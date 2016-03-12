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

#define ELPP_DEFAULT_LOG_FILE "logs/emws-train-%datetime.log"

#include "easylogging++.h"

INITIALIZE_EASYLOGGINGPP


#include "base_seger.h"
#include "utf8_io.h"

void print_config(rapidjson::Document const &config) {

    using namespace std;
    for (auto m = config.MemberBegin();
         m != config.MemberEnd(); ++m) {
        cout << '\t' << m->name.GetString() << ": ";
        if (m->value.IsString())
            cout << m->value.GetString();
        else if (m->value.IsUint())
            cout << m->value.GetUint();
        else if (m->value.IsDouble())
            cout << m->value.GetDouble();
        else if (m->value.IsBool())
            cout << (m->value.GetBool() ? "true" : "false");
        else if (m->value.IsNull())
            cout << "null";
        else
            cout << "unknown type";
        cout << endl;
    }
}

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
                ("config,c", value<string>(), "config file");

        variables_map vm;
        store(parse_command_line(argc, argv, desc), vm);
        notify(vm);

        if (vm.count("help")) {
            cout << desc << endl;
            exit(0);
        }
        if (!vm.count("config")) {
            cerr << "Must specify config file path." << endl;
            cout << desc << endl;
            exit(1);
        }

        string const filename = vm["config"].as<string>();
        mainLogger->info("use config file: %v", filename);

        mainLogger->info("parse parameters ...");
        FILE* fp = fopen(filename.data(), "rb");
        char readBuffer[65536];
        FileReadStream is(fp, readBuffer, sizeof(readBuffer));
        Document config;
        config.ParseStream(is);
        fclose(fp);

        StringBuffer buffer;
        PrettyWriter<StringBuffer> writer(buffer);
        config.Accept(writer);

        // print_config(config);
        // cout << boolalpha << config.IsNull() << endl;
        mainLogger->info("=====parameters====\n%v", buffer.GetString());

        mainLogger->info("initial model ...");
        string const model_name = "emws";
        base_seger *seger = base_seger::create_seger(config, model_name);

        mainLogger->info("train ...");
        seger->train();

        string const model_path = utf8_io::gen_model_path(model_name);
        mainLogger->info("save model to %v", model_path);
        // save model
        seger->save(model_path);

        mainLogger->info("train done");

        delete seger;
    }
    catch (const error &ex)
    {
        std::cerr << ex.what() << '\n';
    }

}