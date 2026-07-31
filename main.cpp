#include <iostream>
#include "dataloader.h"     // the file you just wrote (loadData8bit, LoadData, etc.)
#include "network.h"   // Layer, OutputLayer, Model

using namespace std;

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
    int batch_size = 6000;

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

    // ---- Train ----
    int epochs = 5;
    double learning_rate = 0.01;
    for (int e = 0; e < epochs; e++){
        model.train(learning_rate);
        cout << "Epoch " << e << " last-batch loss: " << model.loss_val << endl;
    }

    // ---- Load test data ----
    Tensor<double,1> test_x;
    Tensor<double,2> test_y;
    LoadData(test_x, test_y,
              "data/t10k-images.idx3-ubyte",
              "data/t10k-labels.idx1-ubyte",
              10);
    test_x = test_x / 255.0;

    // ---- Evaluate accuracy on a batch of test data ----
    int test_batch_size = 100;
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

    destroy();
    return 0;
}