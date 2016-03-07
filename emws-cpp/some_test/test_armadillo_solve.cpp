//
// Created by junfeng on 2/28/16.
//

#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int main()
{
    vec b;
    b << 2.0 << 5.0 << 2.0;

    // endr represents the end of a row
    // in a matrix
    mat A;
    A << 1.0 << 2.0 << endr
    << 2.0 << 3.0 << endr
    << 1.0 << 3.0 << endr;

    cout << "Least squares solution:" << endl;
    cout << solve(A,b) << endl;

    return 0;
}