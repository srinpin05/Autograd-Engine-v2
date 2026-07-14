#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;



//Operations
class Node {
    public:
        vector<Node*> inputs;
        static vector<Node*> visited;
        bool param = false;
        double d_loss = 0;
        int forward_count = 0;
        int backward_count = 0;
        bool visit = false;
        double val;
        virtual void backward(double upstream, bool output = false){}
        virtual void propogate(){} //helper function to push local gradient down to inputs.
        ~Node(){
        }
        void walk(Node* root){
            if (root->visit){return;}
            root->visit = true; 
            for (Node* i : root->inputs){
                walk(i);
            }
            visited.push_back(root);
    
        }
        void reset(){
            d_loss = 0;
            visited.clear();
        }
};
vector<Node*> Node::visited;
class ParamNode : public Node{
    public: 
        ParamNode(int value){
            param = true;
            val = value;
        }
        void backward(double upstream, bool output = false){
            d_loss += upstream;
        }
};


class AddNode : public Node{
    public:
        AddNode(Node& x, Node& y){
            inputs.push_back(&x);
            x.forward_count++;
            inputs.push_back(&y);
            y.forward_count++;
            val = 0;
            for (int i = 0; i<inputs.size();i++){
                val += inputs[i]->val;
            }
        }
        void backward(double upstream, bool output = false){
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
            inputs.push_back(&x);
            x.forward_count++;
            inputs.push_back(&y);
            y.forward_count++;
            val = 1;
            for (int i = 0; i<inputs.size();i++){
                val *= inputs[i]->val;
            }
        }

        void backward(double upstream, bool output = false){
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
            inputs.push_back(&x);
            x.forward_count += 1;
            val = log(x.val);
        }
        void backward(double upstream, bool output = false){
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
            inputs.push_back(&x);
            x.forward_count++;
            val = 1/(1+exp(0-x.val));
        }
        void backward(double upstream, bool output = false){
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
            inputs.push_back(&x);
            x.forward_count++;
            val = (x.val>0) ? x.val : 0;
        }
        void backward(double upstream, bool output = false){
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
            inputs.push_back(&x);
            x.forward_count++;
            val = tanh(x.val);
        }
        void backward(double upstream, bool output = false){
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

int main(){
    ParamNode a(3);
    ParamNode b(4);
    ParamNode c(5);

    Node& d = a+b;
    Node& e = d*b;
    Node& f = d+e;
    f.backward(1, true);
    cout<<b.d_loss;
    destroy();
}
