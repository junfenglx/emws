//
// Created by junfeng on 2/29/16.
//

#ifndef EMWS_CPP_ARMADILLO_BENCH_H
#define EMWS_CPP_ARMADILLO_BENCH_H

#include <armadillo>

#include "base_bench.h"

class armadillo_bench : public base_bench {
public:
    armadillo_bench(int n_cols, int n_rows, int n_times, int n_selected_rows) :
            base_bench(n_cols, n_rows, n_times, n_selected_rows) {}
    virtual ~armadillo_bench() {}
    void create_matrix();
    void run();
    void only_mul();

private:
    arma::mat m_mat;
};


#endif //EMWS_CPP_ARMADILLO_BENCH_H
