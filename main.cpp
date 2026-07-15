#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "autograd.h"
using namespace std;


int main(){
    ParamNode a(3);
    ParamNode b(4);
    ParamNode c(5);

    Node& d = a+b;
    Node& e = d*b;
    Node& f = d+e;
    e.backward(1, true);
    f.backward(1, true);
    cout<<b.d_loss;
    destroy();
}
