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
    // ------------------------------------------------------------------
    // Build a tiny 2-layer network:
    //   input(2) -> hidden(3) -> output(2)
    // ------------------------------------------------------------------

    // 1. Input vector (2 features)
    VectorXd x(2);
    x << 1.0, 0.5;

    // 2. Ground truth (2 targets)
    VectorXd y(2);
    y << 1.0, 0.0;

    Layer* layer0 = new Layer(2, 3, x);
    Layer* layer1 = new Layer(3, 2);

    OutputLayer truth(2, y);

    // 5. Wire everything into a Model.
    Model model;
    model.add_layer(layer0);
    model.add_layer(layer1);
    model.set_truevalues(&truth);

    // 6. Forward pass.
    cout << "=== forward ===\n";
    model.forward();
    cout << "predicted =\n" << layer1->activations->val << "\n\n";

    // 7. Backward pass + print parameter gradients.
    cout << "=== backward ===\n";
    model.backward_propogate();

    // 8. Cleanup. The Layer/OutputLayer-owned ParamNodes are leaked
    //    (we used `new` to allocate them). The forward-pass nodes
    //    allocated via operator*/operator+ are tracked in `global`
    //    and freed by destroy().
    destroy();

    return 0;
}
