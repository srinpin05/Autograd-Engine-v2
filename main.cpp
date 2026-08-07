#include <iostream>
#include "dataloader.h"     // the file you just wrote (loadData8bit, LoadData, etc.)
#include "network.h"   // Layer, OutputLayer, Model

using namespace std;
Tape global_tape;
// main.cpp
Arena param_arena(16 * 1024 * 1024);        // small, permanent
Arena transient_arena(64 * 1024 * 1024);   // per-batch, reset every iteration
#include <fstream>

void save_model(Model& model, const string& path){
    ofstream out(path, ios::binary);
    if (!out.is_open()){ cerr << "Failed to open " << path << " for writing.\n"; return; }

    int num_layers = model.l.size();
    out.write((char*)&num_layers, sizeof(int));

    for (Layer* layer : model.l){
        auto& w = layer->weights.val;
        int wd0 = w.dimension(0), wd1 = w.dimension(1), wd2 = w.dimension(2);
        out.write((char*)&wd0, sizeof(int));
        out.write((char*)&wd1, sizeof(int));
        out.write((char*)&wd2, sizeof(int));
        out.write((char*)w.data(), sizeof(double) * w.size());

        auto& b = layer->bias.val;
        int bd0 = b.dimension(0), bd1 = b.dimension(1), bd2 = b.dimension(2);
        out.write((char*)&bd0, sizeof(int));
        out.write((char*)&bd1, sizeof(int));
        out.write((char*)&bd2, sizeof(int));
        out.write((char*)b.data(), sizeof(double) * b.size());
    }
    out.close();
    cout << "Model saved to " << path << endl;
}
int main(){
    // ---- Load training data ----
    Tensor<double,1> train_x;
    Tensor<double,2> train_y;
    LoadData(train_x, train_y,
              "data/train-images.idx3-ubyte",
              "data/train-labels.idx1-ubyte",
              10);

    // Normalize pixel values from [0,255] to [0,1] — helps training stability a lot
    train_x = train_x / 255.0;

    int input_size = size_of_sample;   // 784 for MNIST (28*28), set by loadData8bit
    int num_classes = 10;
    int batch_size = 32;

    // ---- Build model ----
    Model model(train_x, train_y);

    Layer* layer1 = new Layer(input_size, 128, train_x, batch_size); // input layer
    Layer* layer2 = new Layer(128, 64);                              // hidden layer
    Layer* layer3 = new Layer(64, num_classes);                      // output layer
    model.add_layer(layer1);
    model.add_layer(layer2);
    model.add_layer(layer3);

    OutputLayer* out = new OutputLayer(num_classes, train_y, batch_size);
    model.set_truevalues(out);

    // ---- Train (SGD, one sample at a time) ----
    int epochs = 10;
    ProgressBar bar;
    double learning_rate = 0.01;
    for (int e = 0; e < epochs; e++){
    model.train_sgd(learning_rate, bar, e, epochs);
    }
    save_model(model, "model.bin");
    // Debug: overfit a tiny fixed batch to sanity-check the engine

    // ---- Load test data ----
    Tensor<double,1> test_x;
    Tensor<double,2> test_y;
    LoadData(test_x, test_y,
              "data/t10k-images.idx3-ubyte",
              "data/t10k-labels.idx1-ubyte",
              10);
    test_x = test_x / 255.0;
}