//
// Created by junfeng on 2/29/16.
//

#include <iostream>

#include "armadillo_bench.h"

using std::cout;
using std::endl;

void armadillo_bench::create_matrix() {
    using namespace arma;
    m_mat = randn<mat>(n_rows, n_cols);
    // cout << m_mat << endl;
}

void armadillo_bench::run() {
    using namespace arma;
    for (int i=0; i < n_times; ++i) {
        uvec row_indices = randi < uvec > (n_selected_rows, distr_param(0, m_mat.n_rows - 1));
        mat features = m_mat.rows(row_indices);
        mat factor = randn < mat > (n_cols, n_cols);
        // cout << row_indices << endl;
        // cout << features << endl;
        features *= factor;
        m_mat.rows(row_indices) = features;
        // cout << features << endl;
        // cout << m_mat << endl;
    }
}

void armadillo_bench::only_mul() {
    using namespace arma;
    for (int i=0; i < n_times; ++i) {
        mat factor = randn < mat > (n_cols, n_cols);
        m_mat *= factor;
    }
}
