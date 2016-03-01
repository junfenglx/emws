//
// Created by junfeng on 2/29/16.
//

#ifndef EMWS_CPP_EIGEN_BENCH_H
#define EMWS_CPP_EIGEN_BENCH_H

#include <Eigen/Dense>

#include "base_bench.h"

class eigen_bench : public base_bench {
public:
    eigen_bench(int n_cols, int n_rows, int n_times, int n_selected_rows) :
            base_bench(n_cols, n_rows, n_times, n_selected_rows) {}
    virtual ~eigen_bench() {}
    void create_matrix();
    void run();
    void only_mul();

private:
    Eigen::MatrixXd m_mat;
    // dynamic-size matrix whose size is currently 0-by-0,
    // and whose array of coefficients hasn't yet been allocated at all.
};


#endif //EMWS_CPP_EIGEN_BENCH_H
