//
// Created by junfeng on 3/2/16.
//

// #define ELPP_DEFAULT_LOG_FILE "logs/myeasylog-%datetime{%H:%m}.log"

#define ELPP_DEFAULT_LOG_FILE "logs/test-easylog-%datetime.log"

#include "easylogging++.h"

INITIALIZE_EASYLOGGINGPP

int main(int argc, char *argv[]) {
    START_EASYLOGGINGPP(argc, argv);
    LOG(INFO) << "My first info log using default logger";
    CLOG(ERROR, "performance") << "This is info log using performance logger";
    el::Logger* mainLogger = el::Loggers::getLogger("main");
    mainLogger->debug("Hello, main logger");
    CLOG(ERROR, "main") << "CLOG using main logger";
    return 0;
}