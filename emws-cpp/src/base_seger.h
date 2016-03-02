//
// Created by junfeng on 3/2/16.
//

#ifndef PROJECT_BASE_SEGER_H
#define PROJECT_BASE_SEGER_H

#include <string>

class base_seger {
public:
    virtual void train() = 0;
    virtual void predict(std::string const & test_file) = 0;
};


#endif //PROJECT_BASE_SEGER_H
