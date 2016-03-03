//
// Created by junfeng on 3/2/16.
//

#include <iostream>
#include <string>
#include <cstdio>

#include <boost/program_options.hpp>

#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"

#define ELPP_DEFAULT_LOG_FILE "logs/emws-train-%datetime.log"

#include "easylogging++.h"

INITIALIZE_EASYLOGGINGPP


#include "base_seger.h"


int main(int argc, const char *argv[]) {
    using namespace std;
    using namespace rapidjson;
    using namespace boost::program_options;

    START_EASYLOGGINGPP(argc, argv);
    LOG(INFO) << "My first info log using default logger";
    CLOG(ERROR, "performance") << "This is info log using performance logger";
    el::Logger* mainLogger = el::Loggers::getLogger("main");
    mainLogger->debug("Hello, main logger");
    CLOG(ERROR, "main") << "CLOG using main logger";

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
        cout << filename << endl;
        FILE* fp = fopen(filename.data(), "rb");
        char readBuffer[65536];
        FileReadStream is(fp, readBuffer, sizeof(readBuffer));
        Document config;
        config.ParseStream(is);
        fclose(fp);

        for (auto m = config.MemberBegin();
             m != config.MemberEnd(); ++m) {
            cout << m->name.GetString() << ": ";
            if (m->value.IsString())
                cout << m->value.GetString();
            else if (m->value.IsInt())
                cout << m->value.GetInt();
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
        base_seger *seger = base_seger::create_seger(config, string("emws"));
        seger->train();

        delete seger;
    }
    catch (const error &ex)
    {
        std::cerr << ex.what() << '\n';
    }

}