//
// Created by junfeng on 2/29/16.
//

#include <iostream>


#include <igl/floor.h>
#include <igl/slice.h>
#include <igl/slice_into.h>

#include "eigen_bench.h"

using std::cout;
using std::endl;

void eigen_bench::create_matrix() {
    using namespace Eigen;
    m_mat = MatrixXd::Random(n_rows, n_cols);
    // cout << m_mat << endl;
}

void eigen_bench::run() {
    using namespace Eigen;
    for (int i=0; i < n_times; ++i) {
        VectorXi row_indices;
        igl::floor((0.5 * (VectorXd::Random(n_selected_rows).array() + 1.) * m_mat.rows()).eval(), row_indices);
        MatrixXd features;
        igl::slice(m_mat, row_indices, 1, features);
        // cout << row_indices << endl;
        // cout << features << endl;
        MatrixXd factor = MatrixXd::Random(n_cols, n_cols);
        features = features * factor;
        igl::slice_into(features, row_indices, 1, m_mat);
        // cout << m_mat << endl;
    }
}

void eigen_bench::only_mul() {
    using namespace Eigen;
    for (int i=0; i < n_times; ++i) {
        MatrixXd factor = MatrixXd::Random(n_cols, n_cols);
        m_mat = m_mat * factor;
        // cout << m_mat << endl;
    }
}
