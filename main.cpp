#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "autograd.h"
using namespace std;


int main(){
    Eigen::MatrixXd m2(2, 3);
    m2 << 1.5, 2.5, 3.5,
          4.5, 5.5, 6.5;
    Eigen::MatrixXd m1(3, 1);
    m1 << 1.5,
          2.5,
          3.5;
    ParamNode a(m2);
    ParamNode b(m1);
    Node& d = a*b;
    // seed gradient must match d.val's shape (2x1 here, since m2 is 2x3 and m1 is 3x1)
    Eigen::MatrixXd seed = Eigen::MatrixXd::Ones(d.val.rows(), d.val.cols());
    d.backward(seed);
    cout << "d.val =\n" << d.val << "\n\n";
    cout << "a.d_loss =\n" << a.d_loss << "\n\n";
    cout << "b.d_loss =\n" << b.d_loss << "\n";
    destroy();
}
