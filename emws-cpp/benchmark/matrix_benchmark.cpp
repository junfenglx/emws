#include <iostream>
#include <string>
#include <cstdlib>
#include <chrono>

#include <boost/program_options.hpp>

#include "base_bench.h"
#include "eigen_bench.h"
#include "armadillo_bench.h"

int main(int argc, const char *argv[]) {
    using namespace std;
    using namespace boost::program_options;

    int n_cols = 50;
    int n_rows = 50000;
    int n_times = 50000;
    int n_selected_rows = 12;
    string run_type = "sub_indices";
    try
    {
        options_description desc{"Options"};
        desc.add_options()
                ("help,h", "show help message")
                ("n_cols,c", value<int>()->default_value(n_cols), "n_cols")
                ("n_rows,r", value<int>()->default_value(n_rows), "n_rows")
                ("n_times,t", value<int>()->default_value(n_times), "n_times")
                ("n_selected_rows,s", value<int>()->default_value(n_selected_rows), "n_selected_rows")
                ("bench_lib,b", value<string>(), "benchmark library (eigen, armadillo)")
                ("run_type", value<string>()->default_value(run_type), "run type (sub_indices, only_mul)");

        variables_map vm;
        store(parse_command_line(argc, argv, desc), vm);
        notify(vm);

        if (vm.count("help")) {
            cout << desc << endl;
            exit(0);
        }
        if (!vm.count("bench_lib")) {
            cerr << "Must specify which library to test." << endl;
            cout << desc << endl;
            exit(1);
        }
        n_cols = vm["n_cols"].as<int>();
        n_rows = vm["n_rows"].as<int>();
        n_times = vm["n_times"].as<int>();
        n_selected_rows = vm["n_selected_rows"].as<int>();
        string bench_lib = vm["bench_lib"].as<string>();
        run_type = vm["run_type"].as<string>();

        base_bench *bench = nullptr;

        if (bench_lib == "eigen") {
            bench = new eigen_bench(n_cols, n_rows, n_times, n_selected_rows);
        }
        else if (bench_lib == "armadillo") {
            bench = new armadillo_bench(n_cols, n_rows, n_times, n_selected_rows);
        }
        else {
            cerr << "Unknow library " << bench_lib << endl;
            cout << desc << endl;
            exit(1);
        }
        cout << "n_cols: " << n_cols << endl;
        cout << "n_rows: " << n_rows << endl;
        cout << "n_times: " << n_times << endl;
        cout << "n_selected_rows: " << n_selected_rows << endl;
        cout << "benchmark library: " << bench_lib << endl;
        cout << "run_type: " << run_type << endl;

        std::chrono::time_point<std::chrono::system_clock> start, end;

        start = std::chrono::system_clock::now();
        bench->create_matrix();
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        std::cout << "created matrix used time: " << elapsed_seconds.count() << 's' << endl;

        start = std::chrono::system_clock::now();
        if (run_type == "sub_indices") {
            bench->run();
        }
        else if (run_type == "only_mul") {
            bench->only_mul();
        }
        else {
            cerr << "Unknow run type " << run_type << endl;
            cout << desc << endl;
            exit(1);
        }
        end = std::chrono::system_clock::now();
        elapsed_seconds = end-start;
        std::cout << "run matrix multiplication used time: " << elapsed_seconds.count() << 's' << endl;

        delete bench;
    }
    catch (const error &ex)
    {
        std::cerr << ex.what() << '\n';
    }
    return 0;
}
