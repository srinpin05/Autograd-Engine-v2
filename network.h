#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "autograd.h"
#include <optional>

using namespace std;
using namespace Eigen;

//TODO: Finish Epochs Implementatoin
//TODO: Finish Batches Implementation

// Helper: random weight tensor, shape (1, output_size, input_size) — batch=1 since
// weights are shared across the batch (MultNode/AddNode broadcast batch=1 automatically).
inline Tensor<double,3> random_weights(int output_size, int input_size){
    Tensor<double,3> w(1, output_size, input_size);
    w.setRandom();
    return w;
}

// Helper: random bias tensor, shape (1, output_size, 1)
inline Tensor<double,3> random_bias(int output_size){
    Tensor<double,3> b(1, output_size, 1);
    b.setRandom();
    return b;
}

class Layer {
    public:
        bool input_layer = false;
        bool output_layer = false;
        Node& weights;
        Node& bias;
        Node* activations = nullptr;  // initialized to nullptr; set during forward()
        int activation_size;

        // input: Tensor<double,3> shaped (batch, input_size, 1)
        Layer(int input_size, int output_size, const Tensor<double,3>& input)
            : input_layer(true)
            , weights(*(new ParamNode(random_weights(output_size, input_size))))
            , bias(*(new ParamNode(random_bias(output_size))))
            , activations(new InputNode(input, false))
            , activation_size(input_size)
        {}

        Layer(int input_size, int output_size)
            : weights(*(new ParamNode(random_weights(output_size, input_size))))
            , bias(*(new ParamNode(random_bias(output_size))))
            , activations(nullptr)  // set during forward()
            , activation_size(input_size)
        {}

};

class OutputLayer {
    public:
        bool output_layer = true;
        Node& activations;
        int activation_size;

        // ground_truth: Tensor<double,3> shaped (batch, output_size, 1)
        OutputLayer(int output_size, const Tensor<double,3>& ground_truth)
            : activations(*(new InputNode(ground_truth, false)))
            , activation_size(output_size)
        {}

};

class Model {
    public:
    vector<Layer*> l;
    OutputLayer* truevalues = nullptr;  // set after construction
    Node* predicted_activations = nullptr;
    double loss_val;
    void forward(){
        for (size_t i = 0; i + 1 < l.size(); i++){
            l[i+1]->activations = new ReLUNode(l[i]->weights * (*l[i]->activations) + l[i]->bias);
            l[i+1]->activations->heap_owned = true;
        }
        // Final layer's activations
        predicted_activations = &(l.back()->weights * (*l.back()->activations) + l.back()->bias);
    }

    void backward_propogate(double learning_rate){
        if (!truevalues) {
            cerr << "Model: truevalues not set. Call set_truevalues() first.\n";
            return;
        }
        // Seed shape must match predicted_activations/loss output shape: (1,1,1)
        Tensor<double,3> seed(1,1,1);
        seed.setConstant(1.0);
        Node& l = loss(*predicted_activations, truevalues->activations);
        loss_val = l.val(0,0,0);
        l.backward(seed);
        update_gradient_params(learning_rate);
        destroy_mid();   // free the per-epoch graph (MultNode/AddNode/MSENode)
    }

    Node& loss(Node& predicted, Node& output){
        Node* loss_node = new SoftmaxCCENode(predicted, output);
        loss_node->heap_owned = true;
        return *loss_node;
    }

        void add_layer(Layer* layer){ l.push_back(layer); }
        void set_truevalues(OutputLayer* tv){ truevalues = tv; }
};