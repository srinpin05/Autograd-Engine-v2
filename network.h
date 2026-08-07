#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "autograd.h"
#include <optional>
#include <iomanip>
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
    return w * sqrt(2.0/input_size);
}

// Helper: random bias tensor, shape (1, output_size, 1)
inline Tensor<double,3> random_bias(int output_size){
    Tensor<double,3> b(1, output_size, 1);
    b.setZero();
    return b;
}

inline void load_next_batch_i(
    const Tensor<double, 1>& data,
    Tensor<double, 3>& batch,
    int current_batch,
    int batch_size,
    int size_of_sample
) {
    const int total_samples = data.size() / size_of_sample;
    const int start_sample = current_batch * batch_size;
    const int this_batch_size =
        std::min(batch_size, total_samples - start_sample);

    batch.resize(this_batch_size, size_of_sample, 1);

    for (int b = 0; b < this_batch_size; ++b) {
        const int sample_offset = (start_sample + b) * size_of_sample;

        for (int feature = 0; feature < size_of_sample; ++feature) {
            batch(b, feature, 0) = data(sample_offset + feature);
        }
    }
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

// ---- SGD helpers (NEW) ----
// Build a (1, F, 1) tensor from row `i` of a (N, F) matrix.
inline Tensor<double, 3> row_to_input_tensor(const Tensor<double, 2>& data, int i){
    int cols = data.dimension(1);
    Tensor<double, 3> t(1, cols, 1);
    for (int c = 0; c < cols; c++){
        t(0, c, 0) = data(i, c);
    }
    return t;
}

// Build a (1, C, 1) one-hot tensor from an int label 0..C-1.
inline Tensor<double, 3> label_to_target_tensor(int label, int num_classes){
    Tensor<double, 3> t(1, num_classes, 1);
    t.setZero();
    t(0, label, 0) = 1.0;
    return t;
}

// Linear index → label, for one-hot rows stored in row-major.
inline int argmax_one_hot(const Tensor<double, 2>& y_row){
    int best = 0;
    double best_val = y_row(0, 0);
    int cols = y_row.dimension(1);
    for (int c = 1; c < cols; c++){
        if (y_row(0, c) > best_val){
            best_val = y_row(0, c);
            best = c;
        }
    }
    return best;
}


// PROGRESS BAR CODE


class ProgressBar {
    int bar_width = 30;
public:
    void update(int epoch, int total_epochs, int batch, int total_batches, double loss){
        double frac = (double)(batch + 1) / total_batches;
        int filled = (int)(frac * bar_width);

        cout << "\rEpoch " << (epoch + 1) << "/" << total_epochs << " ["
             << (batch + 1) << "/" << total_batches << "] |";
        for (int i = 0; i < bar_width; i++)
            cout << (i < filled ? '=' : (i == filled ? '>' : ' '));
        cout << "| loss: " << fixed << setprecision(4) << loss << "   " << flush;
    }
    void done(){
        cout << endl;   // move to next line once the epoch finishes
    }
};


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
        OutputLayer(int output_size, Tensor<double, 2>& ground_truth, int batch_size_=32)
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

    //We are assuming outputs is already one-hot encoded. 
    Model(Tensor<double, 1> &inputs, Tensor<double, 2> &outputs) : input_data(inputs), output_data(outputs){} 
    
    Node& loss(Node& predicted, Node& output){
        Node* loss_node = new SoftmaxCCENode(predicted, output);
        loss_node->heap_owned = true;
        return *loss_node;
    }

    void add_layer(Layer* layer){ l.push_back(layer); }
    void set_truevalues(OutputLayer* tv){ truevalues = tv; }

    //forward method that allocates NEW nodes on the heap for every forward pass
    void forward(){
        for (size_t i = 0; i + 1 < l.size(); i++){
                l[i+1]->activations = new ReLUNode(l[i]->weights * (*l[i]->activations) + l[i]->bias);
                l[i+1]->activations->heap_owned = true;
        }
            // Final layer's activations
        predicted_activations = &(l.back()->weights * (*l.back()->activations) + l.back()->bias);
    
    }
    // Backward method that does one pass of backward propogation
    void backward_propogate(double learning_rate){
        // Seed shape must match predicted_activations/loss output shape: (1,1,1)
        Tensor<double,3> seed(1,1,1);
        seed.setConstant(1.0);

        // Perform backwards pass
        Node& l = loss(*predicted_activations, *truevalues->activations);
        loss_val = l.val(0,0,0);
        l.backward(seed);

        // Update the parameter nodes and zero out the d_loss for the nodes. 
        update_gradient_params(learning_rate);
        destroy();
        zero_params();

    }

void train_sgd(double learning_rate, ProgressBar& bar, int epoch, int total_epochs){
    if (l.empty() || !truevalues) {
        cerr << "Model::train_sgd: model is empty or truevalues not set.\n";
        return;
    }

    int batch_size = l[0]->batch_size;
    batches = (output_data.dimension(0) + batch_size - 1) / batch_size;
    cout<<"Training has started..."<<batches<<endl;

    double running_acc = 0.0;   // NEW: running average training accuracy
    int acc_count = 0;          // NEW

    for (int i = 0; i < batches; i++){
        load_next_batch_i(input_data, l[0]->batch, i, batch_size, l[0]->activation_size);
        load_next_batch_o(output_data, truevalues->batch, i, truevalues->batch_size, truevalues->activation_size);
        l[0]->activations       = new InputNode(l[0]->batch);
        truevalues->activations = new InputNode(truevalues->batch);

        forward();

        // NEW: compute accuracy on this batch BEFORE backward_propogate destroys predicted_activations
        int this_batch_size = l[0]->batch.dimension(0);
        int num_classes = predicted_activations->val.dimension(1);
        int correct = 0;
        for (int b = 0; b < this_batch_size; b++){
            int pred = 0;
            double max_val = predicted_activations->val(b, 0, 0);
            for (int c = 1; c < num_classes; c++){
                if (predicted_activations->val(b, c, 0) > max_val){
                    max_val = predicted_activations->val(b, c, 0);
                    pred = c;
                }
            }
            int truth = 0;
            for (int c = 0; c < num_classes; c++){
                if (truevalues->activations->val(b, c, 0) == 1.0){ truth = c; break; }
            }
            if (pred == truth) correct++;
        }
        double batch_acc = 100.0 * correct / this_batch_size;
        running_acc += batch_acc;
        acc_count++;

        backward_propogate(learning_rate);
        transient_arena.reset();
        bar.update(epoch, total_epochs, i, batches, loss_val);

        // NEW: print running average accuracy every 200 batches
        if ((i + 1) % 200 == 0){
            cout << "  [batch " << (i+1) << "/" << batches
                 << "] running train_acc: " << (running_acc / acc_count) << "%" << endl;
            running_acc = 0.0;
            acc_count = 0;
        }
    }
    bar.done();
}
    /*
    Initial mini-batch batched gradient descent training loop 
    using forward and backward_propogate method 
    */
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
            zero_params();
            Node* cur_input  = l[0]->activations;
            Node* cur_output = truevalues->activations;
            l[0]->activations     = nullptr;
            truevalues->activations = nullptr;
            remove_from_global(cur_input);
            delete cur_input;
            remove_from_global(cur_output);
            delete cur_output;
        }
    }

    // Tiny internal helper so the call site above stays tidy.
    static Tensor<double, 3> row_to_input_tensor_from_row(const Tensor<double, 2>& row){
        int cols = row.dimension(1);
        Tensor<double, 3> t(1, cols, 1);
        for (int c = 0; c < cols; c++){
            t(0, c, 0) = row(0, c);
        }
        return t;
    }
};