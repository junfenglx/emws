//
// Created by junfeng on 3/2/16.
//

#include "base_seger.h"
#include "emws_seger.h"

base_seger * base_seger::create_seger(rapidjson::Document const &config, std::string const &seger_name) {
    base_seger *seger;
    if (seger_name == "emws") {
        seger =  new emws_seger(config);
    }
    else {
        std::string msg{"unknown seger name: "};
        msg += seger_name;
        throw msg;
    }
    return seger;
}
