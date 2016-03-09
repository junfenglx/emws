//
// Created by junfeng on 2/28/16.
//

#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int main()
{
    mat A = randu<mat>(4,5);
    mat B = randu<mat>(4,5);
    cout << A*B.t() << endl;

    uvec v;
    v << 1;
    // 1 element
    // override previous value, size not change
    v << 2;
    cout << v << endl;
    uvec v2;
    // size change, 2 elements
    v2 << 1 << 2;
    // re initialize
    v2 << 3;
    cout << v2 << endl;
    return 0;
}
