#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;
/*This version is v1 and it is very simple 
 but goes through the depth of the entire graph for all shared paths.
 MAJOR inefficiency. Need to make sure shared paths are traveled exactly once when backward() is called.  
*/
//Operations
class Node {
    public:
        vector<Node*> inputs;
        bool param = false;
        double val;
        virtual void backward(double upstream){}
};
class ParamNode : public Node{
    public: 
        int d_loss = 0;
        bool tape = true;
        ParamNode(int value, bool tape){
            this->tape = tape;
            param = true;
            val = value;
        }
        void backward(double upstream){
            if (tape==true){d_loss += upstream;}

        }
};


class AddNode : public Node{
    public:
        AddNode(Node& x, Node& y){
            inputs.push_back(&x);
            inputs.push_back(&y);
            val = 0;
            for (int i = 0; i<inputs.size();i++){
                val += inputs[i]->val;
            }
        }
        void backward(double upstream){
            int grad;
            for (int i = 0; i<inputs.size();i++){
                grad = 1;
                inputs[i]->backward(upstream*grad);
            }
        }
};
class MultNode : public Node{
    public:
        MultNode(Node& x, Node& y){
            inputs.push_back(&x);
            inputs.push_back(&y);
            val = 1;
            for (int i = 0; i<inputs.size();i++){
                val *= inputs[i]->val;
            }
        }

        void backward(double upstream){
            int grad;
            for (int i = 0; i<inputs.size();i++){
                grad = 1;
                for (int j = 0; j<inputs.size();j++){
                    if (i!=j){
                        grad *= inputs[j]->val;
                    }
                }
                inputs[i]->backward(upstream*grad);
            }
        }
};

class LogNode : public Node {
    public:
        LogNode(Node& x) {
            inputs.push_back(&x);
            val = log(x.val);
        }
        void backward(double upstream){
            double grad;
            for (int i = 0; i < inputs.size(); i++){
                grad = 1/inputs[i]->val;
                inputs[i]->backward(upstream*grad);
            }
        } 
};

class SigmoidNode : public Node {
    public:
        SigmoidNode(Node& x){
            inputs.push_back(&x);
            val = 1/(1+exp(0-x.val));
        }
        void backward(double upstream){
            double grad;
            for (int i = 0; i < inputs.size(); i++){
                grad = val*(1-val);
                inputs[i]->backward(upstream*grad);
            }
        }
};

class ReLUNode : public Node {
    public:
        ReLUNode(Node& x){
            inputs.push_back(&x);
            val = (x.val>0) ? x.val : 0;
        }
        void backward(double upstream){
            double grad;
            for (int i = 0; i<inputs.size(); i++){
                grad = (inputs[i]->val <= 0) ? 0 : 1;
                inputs[i]->backward(upstream*grad);
            }
        }
};

class tanhNode : public Node {
    public:
        tanhNode(Node& x){
            inputs.push_back(&x);
            val = tanh(x.val);
        }
        void backward(double upstream){
            double grad;
            for (int i = 0; i<inputs.size(); i++){
                grad = 1-val*val;
                inputs[i]->backward(upstream*grad);
            }
        }
};
Node& operator*(Node& x, Node& y){
    Node* x3 = new MultNode(x, y);
    return *x3;
}
Node& operator+(Node& x, Node& y){
    Node* x2 = new AddNode(x, y);
    return *x2;
}

int main(){
    ParamNode a(3, false);
    ParamNode b(4, true);
    ParamNode c(5, true);

    Node& d = a+b;
    Node& e = d*b;
    Node& f = d+e;
    f.backward(1);
    cout<<b.d_loss;
}

