#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;
int main()
{
  Vector3d v(1,2,3);
  Vector3d w(0,1,2);
  cout << "Dot product: " << v.dot(w) << endl;
  double dp = v.adjoint()*w; // automatic conversion of the inner product to a scalar
  cout << "Dot product via a matrix product: " << dp << endl;
  cout << "Cross product:\n" << v.cross(w) << endl;

  /*
   * matrix doesn't have dot method, not like numpy
  Matrix2d mat;
  mat << 1, 2,
         3, 4;
  cout << "Here is mat.dot(mat):\n" << mat.dot(mat) << endl;
  */
}
