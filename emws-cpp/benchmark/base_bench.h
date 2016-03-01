//
// Created by junfeng on 2/29/16.
//

#ifndef EMWS_CPP_BASE_BENCH_H
#define EMWS_CPP_BASE_BENCH_H


class base_bench {
public:
    base_bench(int cols_n, int rows_n, int times_n, int selected_rows_n) :
            n_cols(cols_n),
            n_rows(rows_n),
            n_times(times_n),
            n_selected_rows(selected_rows_n) {}
    virtual void create_matrix() = 0;
    virtual void run() = 0;
    virtual void only_mul() = 0;
    virtual ~base_bench() {}

protected:
    int n_cols;
    int n_rows;
    int n_times;
    int n_selected_rows;
};


#endif //EMWS_CPP_BASE_BENCH_H
