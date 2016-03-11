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

    cout << "test resize" << endl;
    // according to doc, resize may has bug
    mat C{1, 2, 3, 4};
    cout << C << endl;
    C.resize(2, 2);
    cout << C << endl;

    cout << "test reshape" << endl;
    mat D{1, 2, 3, 4};
    cout << D << endl;
    D.reshape(2, 2);
    cout << D.t() << endl;

    return 0;
}
