//
// Created by junfeng on 3/2/16.
//

#ifndef PROJECT_EMWS_SEGER_H
#define PROJECT_EMWS_SEGER_H

#include "base_seger.h"
#include "rapidjson/document.h"

class emws_seger : public base_seger {


public:
    emws_seger(rapidjson::Document const &config);

    virtual void train() override;

    virtual void predict(std::vector<std::string> const &corpus) const override;

    virtual void predict(std::string const &paragraph) const override;

    virtual ~emws_seger() {}

};


#endif //PROJECT_EMWS_SEGER_H
