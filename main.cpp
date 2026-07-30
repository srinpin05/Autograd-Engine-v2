#include "network.h"
#include <iostream>

int main(){
    // Simple linear regression test: y = 2*x1 - 3*x2 + 1
    // (Model has no activation nonlinearity between layers, so keep the target linear)
    int batch = 4;
    Tensor<double,3> X(batch, 2, 1);
    Tensor<double,3> Y(batch, 1, 1);

    double x1[4] = {1, 2, 3, 4};
    double x2[4] = {0.5, 1, 1.5, 2};
    for (int b = 0; b < batch; b++){
        X(b,0,0) = x1[b];
        X(b,1,0) = x2[b];
        Y(b,0,0) = 2*x1[b] - 3*x2[b] + 1;
    }

    Model model;
    Layer* layer1 = new Layer(2, 1, X);   // single linear layer, input_size=2, output_size=1
    model.add_layer(layer1);

    OutputLayer* out = new OutputLayer(1, Y);
    model.set_truevalues(out);

    int epochs = 2000;
    double lr = 0.01;
    //Train loop
    for (int e = 0; e < epochs; e++){
        zero_params();          // REQUIRED: param d_loss accumulates otherwise
        model.forward();


        model.backward_propogate(lr);

        if (e % 200 == 0){
            cout << "Epoch " << e << " loss: " << model.loss_val << endl;
        }

    }

    // Test: run forward once more and print predictions vs ground truth
    model.forward();
    cout << "\nFinal predictions vs targets:\n";
    for (int b = 0; b < batch; b++){
        cout << "  pred=" << model.predicted_activations->val(b,0,0)
             << "  true=" << Y(b,0,0) << endl;
    }

    destroy();         // free the post-training graph nodes (MultNode/AddNode)
    delete layer1;     // frees the Layer's weights/bias/activations ParamNode+InputNode
    delete out;        // frees the OutputLayer's ground-truth InputNode

    return 0;
}