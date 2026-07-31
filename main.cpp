#include "network.h"
#include <iostream>

int main(){
    // Simple linear regression test: y = 2*x1 - 3*x2 + 1
    // (Model has no activation nonlinearity between layers, so keep the target linear)
    int batch = 4;
    Tensor<double,3> X(batch, 2, 1);
    Tensor<double,3> Y(batch, 3, 1);   // 3 classes (not 1) — one-hot per sample

    /*
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
    */

    X(0,0,0)=0; X(0,1,0)=0; Y(0,0,0)=1; Y(0,1,0)=0; Y(0,2,0)=0;  // class 0
    X(1,0,0)=0; X(1,1,0)=1; Y(1,0,0)=0; Y(1,1,0)=1; Y(1,2,0)=0;  // class 1
    X(2,0,0)=1; X(2,1,0)=0; Y(2,0,0)=0; Y(2,1,0)=0; Y(2,2,0)=1;  // class 2
    X(3,0,0)=1; X(3,1,0)=1; Y(3,0,0)=1; Y(3,1,0)=0; Y(3,2,0)=0;  // class 0


    Model model;
    Layer* l0 = new Layer(2, 4, X);   // hidden: 2 -> 4
    Layer* l1 = new Layer(4, 3);      // output: 4 -> 3
    model.add_layer(l0);
    model.add_layer(l1);

    OutputLayer* out = new OutputLayer(3, Y);
    model.set_truevalues(out);
    int epochs = 5000;
    double lr = 0.1;
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
    Tensor<double, 3> softmax_predictions = softmax(model.predicted_activations->val);
    for (int b = 0; b < batch; b++){
        cout << "  pred=" << softmax_predictions(b,0,0)
             << "  true=" << Y(b,0,0) << endl;
    }

    destroy();         // free the post-training graph nodes (MultNode/AddNode)
    delete l0; delete l1; delete out;   

    return 0;
}