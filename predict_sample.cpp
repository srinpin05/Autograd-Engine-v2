#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "dataloader.h"
#include "network.h"

using namespace std;

// autograd.h declares these globals as extern; each executable owns one set.
Tape global_tape;
Arena param_arena(16 * 1024 * 1024);
Arena transient_arena(64 * 1024 * 1024);

struct LoadedParams {
    Tensor<double, 3> weights;
    Tensor<double, 3> bias;
};

static bool load_layer_params(ifstream& in, LoadedParams& params) {
    int wd0, wd1, wd2;
    if (!in.read(reinterpret_cast<char*>(&wd0), sizeof(int)) ||
        !in.read(reinterpret_cast<char*>(&wd1), sizeof(int)) ||
        !in.read(reinterpret_cast<char*>(&wd2), sizeof(int))) {
        return false;
    }

    params.weights.resize(wd0, wd1, wd2);
    if (!in.read(reinterpret_cast<char*>(params.weights.data()),
                 sizeof(double) * params.weights.size())) {
        return false;
    }

    int bd0, bd1, bd2;
    if (!in.read(reinterpret_cast<char*>(&bd0), sizeof(int)) ||
        !in.read(reinterpret_cast<char*>(&bd1), sizeof(int)) ||
        !in.read(reinterpret_cast<char*>(&bd2), sizeof(int))) {
        return false;
    }

    params.bias.resize(bd0, bd1, bd2);
    return static_cast<bool>(in.read(reinterpret_cast<char*>(params.bias.data()),
                                     sizeof(double) * params.bias.size()));
}

static bool load_model_into(Model& model, const string& path) {
    ifstream in(path, ios::binary);
    if (!in) {
        cerr << "Could not open model: " << path << '\n';
        return false;
    }

    int layer_count;
    if (!in.read(reinterpret_cast<char*>(&layer_count), sizeof(int)) ||
        layer_count != static_cast<int>(model.l.size())) {
        cerr << "Model architecture does not match the 784 -> 128 -> 64 -> 10 network.\n";
        return false;
    }

    for (Layer* layer : model.l) {
        LoadedParams params;
        if (!load_layer_params(in, params)) {
            cerr << "Model file ended unexpectedly.\n";
            return false;
        }
        layer->weights.val = params.weights;
        layer->bias.val = params.bias;
    }
    return true;
}

static bool write_image_svg(const Tensor<double, 3>& image, const string& path) {
    ofstream out(path);
    if (!out) return false;

    constexpr int image_size = 28;
    constexpr int pixel_size = 12;
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << image_size * pixel_size
        << "\" height=\"" << image_size * pixel_size << "\" viewBox=\"0 0 "
        << image_size * pixel_size << ' ' << image_size * pixel_size << "\">\n";
    out << "<rect width=\"100%\" height=\"100%\" fill=\"black\"/>\n";

    for (int row = 0; row < image_size; ++row) {
        for (int column = 0; column < image_size; ++column) {
            const double normalized = image(0, row * image_size + column, 0);
            const int intensity = std::clamp(static_cast<int>(round(normalized * 255.0)), 0, 255);
            out << "<rect x=\"" << column * pixel_size << "\" y=\"" << row * pixel_size
                << "\" width=\"" << pixel_size << "\" height=\"" << pixel_size
                << "\" fill=\"rgb(" << intensity << ',' << intensity << ',' << intensity << ")\"/>\n";
        }
    }
    out << "</svg>\n";
    return true;
}

int main(int argc, char** argv) {
    int sample_index = 0;
    if (argc > 1) {
        try {
            sample_index = stoi(argv[1]);
        } catch (const exception&) {
            cerr << "Usage: " << argv[0] << " [test-sample-index]\n";
            return 1;
        }
    }
    if (sample_index < 0 || sample_index >= 10000) {
        cerr << "Test-sample index must be between 0 and 9999.\n";
        return 1;
    }

    Tensor<double, 1> test_x;
    Tensor<double, 2> test_y;
    LoadData(test_x, test_y,
             "data/t10k-images.idx3-ubyte",
             "data/t10k-labels.idx1-ubyte", 10);
    test_x = test_x / 255.0;

    // A batch of one is intentional: it makes the displayed image, label, and
    // prediction refer to exactly the same test example.
    Model model(test_x, test_y);
    auto* layer1 = new Layer(size_of_sample, 128, test_x, 1);
    auto* layer2 = new Layer(128, 64);
    auto* layer3 = new Layer(64, 10);
    model.add_layer(layer1);
    model.add_layer(layer2);
    model.add_layer(layer3);
    model.set_truevalues(new OutputLayer(10, test_y, 1));

    if (!load_model_into(model, "model.bin")) return 1;

    Tensor<double, 3> image;
    Tensor<double, 3> target;
    load_next_batch_i(test_x, image, sample_index, 1, size_of_sample);
    load_next_batch_o(test_y, target, sample_index, 1, 10);

    layer1->activations = new InputNode(image, false);
    model.forward();

    Tensor<double, 3> probabilities;
    softmax(model.predicted_activations->val, probabilities);
    int predicted_class = 0;
    int true_class = 0;
    for (int c = 1; c < 10; ++c) {
        if (probabilities(0, c, 0) > probabilities(0, predicted_class, 0)) {
            predicted_class = c;
        }
    }
    for (int c = 0; c < 10; ++c) {
        if (target(0, c, 0) == 1.0) true_class = c;
    }

    const string image_path = "prediction_sample_" + to_string(sample_index) + ".svg";
    if (!write_image_svg(image, image_path)) {
        cerr << "Could not write image: " << image_path << '\n';
        return 1;
    }

    cout << "Test sample " << sample_index << '\n'
         << "Image written to: " << image_path << '\n'
         << "Model prediction: " << predicted_class
         << " (confidence " << fixed << setprecision(2)
         << probabilities(0, predicted_class, 0) * 100.0 << "%)\n"
         << "True classification: " << true_class << '\n';

    destroy();
    return 0;
}
