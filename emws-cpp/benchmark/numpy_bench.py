#!/usr/bin/env python
# encoding: utf-8

import argparse
import time

import numpy as np


class numpy_bench(object):
    def __init__(self, n_cols, n_rows, n_times, n_selected_rows):
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.n_times = n_times
        self.n_selected_rows = n_selected_rows
        self.m_mat = None

    def create_matrix(self):
        self.m_mat = np.random.randn(self.n_rows, self.n_cols)

    def run(self):
        for _ in range(self.n_times):
            row_indices = np.random.randint(0, self.n_rows, size=self.n_selected_rows)
            features = self.m_mat[row_indices]
            factor = np.random.randn(self.n_cols, self.n_cols)
            features = np.dot(features, factor)
            self.m_mat[row_indices] = features

    def only_mul(self):
        for _ in range(self.n_times):
            factor = np.random.randn(self.n_cols, self.n_cols)
            self.m_mat = np.dot(self.m_mat, factor)


if __name__ == "__main__":
    n_cols = 50
    n_rows = 50000
    n_times = 50000
    n_selected_rows = 12
    run_type = "sub_indices"

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cols", "-c", type=int, default=n_cols, help="n_cols")
    parser.add_argument("--n_rows", "-r", type=int, default=n_rows, help="n_rows")
    parser.add_argument("--n_times", "-t", type=int, default=n_times, help="n_times")
    parser.add_argument("--n_selected_rows", "-s", type=int, default=n_selected_rows, help="n_selected_rows")
    parser.add_argument("--run_type", default=run_type, help="run type (sub_indices, only_mul)")

    args = parser.parse_args()
    print(args)
    n_cols = args.n_cols
    n_rows = args.n_rows
    n_times = args.n_times
    n_selected_rows = args.n_selected_rows
    run_type = args.run_type

    bench = numpy_bench(n_cols, n_rows, n_times, n_selected_rows)
    start = time.time()
    bench.create_matrix()
    end = time.time()
    print("create matrix used time: {0}{1}".format(end - start, "s"))
    start = time.time()
    if run_type == "sub_indices":
        bench.run()
    elif run_type == "only_mul":
        bench.only_mul()
    else:
        raise Exception("Unknown run_type " + run_type)
    end = time.time()
    print("run matrix multiplication used time: {0}{1}".format(end - start, "s"))

