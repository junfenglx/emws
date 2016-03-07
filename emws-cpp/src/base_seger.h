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

    std::vector<std::vector<std::u32string>> predict(std::vector<std::u32string> const &sentences);

    virtual std::vector<std::u32string> predict(std::u32string const &sentence) const = 0;

    virtual bool save(std::string const &model_path) const = 0;

    virtual ~base_seger() {}

    static base_seger * create_seger(rapidjson::Document const &config, std::string const &seger_name);
    static base_seger * load(std::string const &model_path, std::string const &seger_name);

};


#endif //PROJECT_BASE_SEGER_H
