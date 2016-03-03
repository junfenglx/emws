//
// Created by junfeng on 3/2/16.
//

#ifndef PROJECT_BASE_SEGER_H
#define PROJECT_BASE_SEGER_H

#include <string>
#include <vector>

#include "rapidjson/document.h"

class base_seger {


public:
    virtual void train() = 0;

    virtual void predict(std::vector<std::string> const &corpus) const = 0;

    virtual void predict(std::string const &paragraph) const = 0;

    virtual ~base_seger() {}

    static base_seger * create_seger(rapidjson::Document const &config, std::string const &seger_name);

};


#endif //PROJECT_BASE_SEGER_H
