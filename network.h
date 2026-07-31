#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "autograd.h"
#include <optional>
#include <cassert> 
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

inline void load_next_batch_i(const Tensor<double, 1>& data, Tensor<double, 3> &batch, int current_batch, int batch_size, int size_of_sample){
    int total_samples = data.size() / size_of_sample;
    int start_sample = current_batch * batch_size;
    int this_batch_size = std::min(batch_size, total_samples - start_sample);

    Eigen::array<Eigen::Index,1> start_offsets = { start_sample * size_of_sample };
    Eigen::array<Eigen::Index,1> extents = { this_batch_size * size_of_sample };

    batch = data.slice(start_offsets, extents)
        .reshape(Eigen::array<Eigen::Index,3>{ this_batch_size, size_of_sample, 1 });
}

inline void load_next_batch_o(const Tensor<double, 2>& data, Tensor<double, 3> &batch, int current_batch, int batch_size, int num_classes){
    int total_samples = data.dimension(0);
    int start_sample = current_batch * batch_size;
    int this_batch_size = std::min(batch_size, total_samples - start_sample);

    Eigen::array<Eigen::Index,2> start_offsets = { start_sample, 0 };
    Eigen::array<Eigen::Index,2> extents = { this_batch_size, num_classes };

    batch = data.slice(start_offsets, extents)
        .reshape(Eigen::array<Eigen::Index,3>{ this_batch_size, num_classes, 1 });
}

class Layer {
    public:
        bool input_layer = false;
        bool output_layer = false;
        Node& weights;
        Node& bias;
        Node* activations = nullptr;  // initialized to nullptr; set during forward()
        int activation_size;
        int batch_size;
        Tensor<double, 3> batch;
        // input: Tensor<double,3> shaped (batch, input_size, 1)
        Layer(int input_size, int output_size, Tensor<double, 1>& input, int batch_size_ = 32)
            : input_layer(true)
            , weights(*(new ParamNode(random_weights(output_size, input_size))))
            , bias(*(new ParamNode(random_bias(output_size))))
            , activation_size(input_size)
            , batch_size(batch_size_)
        {
            load_next_batch_i(input, batch, 0, batch_size, input_size);
            activations = new InputNode(batch, false);
        }

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
        Node* activations;
        int activation_size;
        Tensor<double, 3> batch;
        int batch_size; 
        // ground_truth: Tensor<double,3> shaped (batch, output_size, 1)
        OutputLayer(int output_size, Tensor<double, 2>& ground_truth, int batch_size_)
            :activation_size(output_size), batch_size(batch_size_)
        {
            load_next_batch_o(ground_truth, batch, 0, batch_size, output_size);
            activations = new InputNode(batch, false);
        }

};

class Model {
    public:
    vector<Layer*> l;
    int current_batch = 0;
    OutputLayer* truevalues = nullptr;  // set after construction
    Node* predicted_activations = nullptr;
    double loss_val;
    int batches;
    Tensor<double, 1> &input_data;
    Tensor<double, 2> &output_data;
    Model(Tensor<double, 1> &inputs, Tensor<double, 2> &outputs) : input_data(inputs), output_data(outputs){} 
    void forward(){
        for (size_t i = 0; i + 1 < l.size(); i++){
                l[i+1]->activations = new ReLUNode(l[i]->weights * (*l[i]->activations) + l[i]->bias);
                l[i+1]->activations->heap_owned = true;
        }
            // Final layer's activations
        predicted_activations = &(l.back()->weights * (*l.back()->activations) + l.back()->bias);
    
    }

    void train(double learning_rate){
        assert(l[0]->batch_size == truevalues->batch_size);
        if (!truevalues) {
            cerr << "Model: truevalues not set. Call set_truevalues() first.\n";
            return;
        }
        batches = (output_data.dimension(0) + truevalues->batch_size - 1) / truevalues->batch_size;;
        for (int i = 0; i<batches; i++){
            if (i>=1) 
            {
                load_next_batch_i(input_data, l[0]->batch, i, l[0]->batch_size, l[0]->activation_size);
                Node* old_input = l[0]->activations;
                l[0]->activations = new InputNode(l[0]->batch);
                remove_from_global(old_input);
                delete old_input;


                load_next_batch_o(output_data, truevalues->batch, i, truevalues->batch_size, truevalues->activation_size);
                Node* old_output = truevalues->activations;
                truevalues->activations = new InputNode(truevalues->batch);
                remove_from_global(old_output);
                delete old_output;
            }
            forward();
            backward_propogate(learning_rate);
        }
    }
    
    void backward_propogate(double learning_rate){
        // Seed shape must match predicted_activations/loss output shape: (1,1,1)
        Tensor<double,3> seed(1,1,1);
        seed.setConstant(1.0);
        Node& l = loss(*predicted_activations, *truevalues->activations);
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