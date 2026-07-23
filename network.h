#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "autograd.h"
#include <optional>


using namespace std;
using namespace Eigen;

//TODO 1: CREATE LOSS FUNCTION NODES IN autograd.h
//TODO 2: CREATE SOFTMAX IN autograd.h
//TODO 3: CREATE LOSS FUNCTION IN network.h
//TODO 4: CHECK IF VectorXd and MatrixXd are compatible. 

class Model {
    vector<Layer*> l; 
    void forward(){
        for (int i = 0; i<l.size()-1;i++){
            l[i+1]->activations = l[i]->weights*l[i]->activations + l[i]->bias;
        }
    }
    void backward_propogate(){
        MatrixXd seed = VectorXd::Ones(l[l.size()-1]->activation_size);
        loss(l[l.size()-1]->activations).backward(seed);
    }
    Node& loss(Node& output){
        
    }
};

class Layer {
    public:
        bool input_layer = false;
        bool output_layer = false;
        int activation_size;
        Node& weights; 
        Node& bias;
        Node& activations;
        Layer(int input_size, int output_size, MatrixXd input) : 
            input_layer(true),
            //l.push_back(this);
            weights(*(new ParamNode(MatrixXd::Random(output_size, input_size)))),
            bias(*(new ParamNode(VectorXd::Random(output_size)))),
            activations(*(new ParamNode(input))),
            activation_size(input_size)

        {}
        Layer(int input_size, int output_size) :
            weights(*(new ParamNode(MatrixXd::Random(output_size, input_size)))),
            bias(*(new ParamNode(VectorXd::Random(output_size)))),
            activations(*(new ParamNode(VectorXd::Random(input_size)))),
            activation_size(input_size)
        {}


};

class OutputLayer{
    public:
        bool output_layer = true;
        Node& activations;
        int activation_size;
        OutputLayer(int output_size) :
            activations(*(new ParamNode(VectorXd::Random(output_size)))),
            activation_size(output_size)
        {}

};