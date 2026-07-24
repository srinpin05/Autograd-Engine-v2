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

class Layer {
    public:
        bool input_layer = false;
        bool output_layer = false;
        Node& weights;
        Node& bias;
        Node* activations = nullptr;  // initialized to nullptr; set during forward()
        int activation_size;
        Layer(int input_size, int output_size, const MatrixXd& input)
            : input_layer(true)
            , weights(*(new ParamNode(MatrixXd::Random(output_size, input_size))))
            , bias(*(new ParamNode(VectorXd::Random(output_size))))
            , activations(new ParamNode(input))
            , activation_size(input_size)
        {}

        Layer(int input_size, int output_size)
            : weights(*(new ParamNode(MatrixXd::Random(output_size, input_size))))
            , bias(*(new ParamNode(VectorXd::Random(output_size))))
            , activations(nullptr)  // set during forward()
            , activation_size(input_size)
        {}
};

class OutputLayer {
    public:
        bool output_layer = true;
        Node& activations;
        int activation_size;

        OutputLayer(int output_size, const VectorXd& ground_truth)
            : activations(*(new ParamNode(ground_truth)))
            , activation_size(output_size)
        {}
};

class Model {
    public:
    vector<Layer*> l;
    OutputLayer* truevalues = nullptr;  // set after construction
    Node* predicted_activations = nullptr;

    void forward(){
        for (size_t i = 0; i + 1 < l.size(); i++){
            l[i+1]->activations = &(l[i]->weights * (*l[i]->activations) + l[i]->bias);
        }
        // Final layer's activations
        predicted_activations = &(l.back()->weights * (*l.back()->activations) + l.back()->bias);
    }

    void backward_propogate(){
        if (!truevalues) {
            cerr << "Model: truevalues not set. Call set_truevalues() first.\n";
            return;
        }
        // Seed shape must match predicted_activations shape (a column vector)
        MatrixXd seed = MatrixXd::Ones(1,1);
        loss(*predicted_activations, truevalues->activations).backward(seed);
        update_gradient_params(0.01);
    }

    Node& loss(Node& output, Node& predicted){
        Node* loss_node = new MSENode(output, predicted);
        return *loss_node;
    }

        void add_layer(Layer* layer){ l.push_back(layer); }
        void set_truevalues(OutputLayer* tv){ truevalues = tv; }
};
