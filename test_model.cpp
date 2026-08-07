#include <iostream>
#include <fstream>
#include "dataloader.h"
#include "network.h"

using namespace std;

// autograd.h declares these extern — every binary that links autograd.h needs its own definitions
Tape global_tape;
Arena param_arena(16 * 1024 * 1024);
Arena transient_arena(64 * 1024 * 1024);

struct LoadedParams { Tensor<double,3> w, b; };

LoadedParams load_layer_params(ifstream& in){
    int wd0, wd1, wd2;
    in.read((char*)&wd0, sizeof(int));
    in.read((char*)&wd1, sizeof(int));
    in.read((char*)&wd2, sizeof(int));
    Tensor<double,3> w(wd0, wd1, wd2);
    in.read((char*)w.data(), sizeof(double) * w.size());

    int bd0, bd1, bd2;
    in.read((char*)&bd0, sizeof(int));
    in.read((char*)&bd1, sizeof(int));
    in.read((char*)&bd2, sizeof(int));
    Tensor<double,3> b(bd0, bd1, bd2);
    in.read((char*)b.data(), sizeof(double) * b.size());

    return {w, b};
}

void load_model_into(Model& model, const string& path){
    ifstream in(path, ios::binary);
    if (!in.is_open()){ cerr << "Failed to open " << path << " for reading.\n"; return; }

    int num_layers;
    in.read((char*)&num_layers, sizeof(int));
    if (num_layers != (int)model.l.size()){
        cerr << "Layer count mismatch: file has " << num_layers
             << ", model expects " << model.l.size() << endl;
        return;
    }
    for (Layer* layer : model.l){
        LoadedParams p = load_layer_params(in);
        layer->weights.val = p.w;
        layer->bias.val = p.b;
    }
    in.close();
    cout << "Model loaded from " << path << endl;
}

int main(){
    Tensor<double,1> test_x;
    Tensor<double,2> test_y;
    LoadData(test_x, test_y, "data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", 10);
    test_x = test_x / 255.0;

    int input_size = size_of_sample;
    int num_classes = 10;
    int batch_size = 32; // must match architecture used in main.cpp

    // Rebuild the exact same architecture — sizes must match training
    Model model(test_x, test_y);
    Layer* layer1 = new Layer(input_size, 128, test_x, batch_size);
    Layer* layer2 = new Layer(128, 64);
    Layer* layer3 = new Layer(64, num_classes);
    model.add_layer(layer1);
    model.add_layer(layer2);
    model.add_layer(layer3);

    OutputLayer* out = new OutputLayer(num_classes, test_y, batch_size);
    model.set_truevalues(out);

    load_model_into(model, "model.bin");   // overwrite the random init with trained weights

    int test_batch_size = 10000;
    Tensor<double,3> x_batch, y_batch;
    load_next_batch_i(test_x, x_batch, 0, test_batch_size, input_size);
    load_next_batch_o(test_y, y_batch, 0, test_batch_size, num_classes);

    Node* test_input = new InputNode(x_batch, false);
    layer1->activations = test_input;
    model.forward();

    int correct = 0;
    for (int b = 0; b < test_batch_size; b++){
        int predicted_class = 0;
        double max_val = model.predicted_activations->val(b, 0, 0);
        for (int c = 1; c < num_classes; c++){
            if (model.predicted_activations->val(b, c, 0) > max_val){
                max_val = model.predicted_activations->val(b, c, 0);
                predicted_class = c;
            }
        }
        int true_class = 0;
        for (int c = 0; c < num_classes; c++){
            if (y_batch(b, c, 0) == 1.0){ true_class = c; break; }
        }
        if (predicted_class == true_class) correct++;
    }

    cout << "Test accuracy on " << test_batch_size << " samples: "
         << (100.0 * correct / test_batch_size) << "%" << endl;
}