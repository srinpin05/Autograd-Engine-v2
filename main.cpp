#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "autograd.h"
#include "network.h"

using namespace std;
using namespace Eigen;

int main(){

    VectorXd x(2);
    x << 1.0, 0.5;


    VectorXd y(2);
    y << 1.0, 0.0;

    Layer* layer0 = new Layer(2, 3, x);
    Layer* layer1 = new Layer(3, 2);

    OutputLayer truth(2, y);


    Model model;
    model.add_layer(layer0);
    model.add_layer(layer1);
    model.set_truevalues(&truth);


    cout << "=== forward ===\n";
    model.forward();
    cout << "predicted =\n" << layer1->activations->val << "\n\n";


    cout << "=== backward ===\n";
    model.backward_propogate();


    destroy();

    return 0;
}