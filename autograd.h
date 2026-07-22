#ifndef AUTOGRAD_H
#define AUTOGRAD_H


#include <cmath>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

class Node {
    public:
        vector<Node*> inputs;
        vector<Node*> visited;
        bool param = false;
        double d_loss = 0;
        int forward_count = 0;
        int backward_count = 0;
        bool visit = false;
        bool requires_grad = true; 
        double val;
        virtual void backward(double upstream, bool output = false){}
        virtual void propogate(){} //helper function to push local gradient down to inputs.
        virtual ~Node(){}
        void walk(Node* root){
            if (root->visit){return;}
            root->visit = true; 
            for (Node* i : root->inputs){
                walk(i);
            }
            visited.push_back(root);
    
        }
        void reset(){
            for(Node* node : visited){
                node->d_loss = 0;
                node->visit = false;
            }
            visited.clear();
        }
};
class ParamNode : public Node{
    public: 
        ParamNode(double value, bool requires_grad = true){
            param = true;
            val = value;
            this->requires_grad = requires_grad;
        }
        void backward(double upstream, bool output = false){
            d_loss += upstream;
        }
};


class AddNode : public Node{
    public:
        AddNode(Node& x, Node& y){
            if (x.requires_grad) inputs.push_back(&x);
            if (y.requires_grad) inputs.push_back(&y);
            val = 0;
            for (int i = 0; i<inputs.size();i++){
                val += inputs[i]->val;
            }
        }
        void backward(double upstream, bool output = false){
            reset();
            double grad;
            d_loss += upstream;
            walk(this);
            reverse(visited.begin(), visited.end());
            for (Node* node : visited){
                node->propogate();
            }
        }
        void propogate(){
            double grad;
            for (int i = 0; i<inputs.size();i++){
                grad = 1;
                inputs[i]->d_loss += d_loss*grad;
            }
        }
};

class MultNode : public Node{
    public:
        MultNode(Node& x, Node& y){
            if (x.requires_grad) inputs.push_back(&x);
            if (y.requires_grad) inputs.push_back(&y);
            y.forward_count++;
            val = 1;
            for (int i = 0; i<inputs.size();i++){
                val *= inputs[i]->val;
            }
        }

        void backward(double upstream, bool output = false){
            reset();
            d_loss += upstream;
            double grad;
            walk(this);
            reverse(visited.begin(), visited.end());
            for (Node* node : visited){
                node->propogate();
            }
        }
        void propogate(){
            double grad;
            for (int i = 0; i<inputs.size();i++){
                grad = 1;
                for (int j = 0; j<inputs.size();j++){
                    if (i!=j){
                        grad *= inputs[j]->val;
                    }
                }
                inputs[i]->d_loss += d_loss*grad;
            }
        }
};

class LogNode : public Node {
    public:
        LogNode(Node& x) {
            if (x.requires_grad) inputs.push_back(&x);
            val = log(x.val);
        }
        void backward(double upstream, bool output = false){
            reset();
            d_loss+=upstream;
            walk(this);
            reverse(visited.begin(), visited.end());
            for(Node* node : visited){
                node->propogate();
            }
        } 
        void propogate(){
            double grad;
            for (int i = 0; i < inputs.size(); i++){
                grad = 1/inputs[i]->val;
                inputs[i]->d_loss += d_loss*grad;
            }
        }
};

class SigmoidNode : public Node {
    public:
        SigmoidNode(Node& x){
            if (x.requires_grad) inputs.push_back(&x);
            val = 1/(1+exp(0-x.val));
        }
        void backward(double upstream, bool output = false){
            reset();
            d_loss += upstream;
            walk(this);
            reverse(visited.begin(), visited.end());
            for (Node* node : visited){
                node->propogate();
            }
        }
        void propogate(){
            double grad;
            for (int i = 0; i < inputs.size(); i++){
                grad = val*(1-val);
                inputs[i]->d_loss += d_loss*grad;
            }
        }
};

class ReLUNode : public Node {
    public:
        ReLUNode(Node& x){
            if (x.requires_grad) inputs.push_back(&x);
            val = (x.val>0) ? x.val : 0;
        }
        void backward(double upstream, bool output = false){
            reset();
            d_loss += upstream;
            walk(this);
            reverse(visited.begin(), visited.end());
            for (Node* node : visited){
                node->propogate();
            }
        }
        void propogate(){
            double grad;
            for (int i = 0; i<inputs.size(); i++){
                grad = (inputs[i]->val <= 0) ? 0 : 1;
                inputs[i]->d_loss += d_loss*grad;
            }
        }
};

class tanhNode : public Node {
    public:
        tanhNode(Node& x){
            if (x.requires_grad) inputs.push_back(&x);
            val = tanh(x.val);
        }
        void backward(double upstream, bool output = false){
            reset();
            d_loss+=upstream;
            walk(this);
            reverse(visited.begin(), visited.end());
            for(Node* node : visited){
                node->propogate();
            }
        }
        void propogate(){
            double grad;
            for (int i = 0; i<inputs.size(); i++){
                grad = 1-val*val;
                inputs[i]->d_loss += d_loss*grad;
            }
        }
};


vector<Node*> global;
Node& operator*(Node& x, Node& y){
    Node* x3 = new MultNode(x, y);
    global.push_back(x3);
    return *x3;
}
Node& operator+(Node& x, Node& y){
    Node* x2 = new AddNode(x, y);
    global.push_back(x2);
    return *x2;
}

void destroy(){
    for (Node* node : global){
        delete node;
    }
    global.clear();
}
#endif