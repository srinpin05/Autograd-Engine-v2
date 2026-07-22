#ifndef AUTOGRAD_H
#define AUTOGRAD_H


#include <cmath>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
// REMEMBER TO CALL zero_params between each backwards() call. 
class Node {
    public:
        vector<Node*> inputs;
        vector<Node*> visited;
        bool param = false;
        MatrixXd d_loss;
        int forward_count = 0;
        int backward_count = 0;
        bool visit = false;
        bool requires_grad = true; 
        MatrixXd val;
        virtual void backward(MatrixXd upstream){}
        virtual void propogate(){} //helper function to push local gradient down to inputs.
        virtual ~Node(){}
        void init_grad_buffer(){
            d_loss = MatrixXd::Zero(val.rows(), val.cols());
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
            for(Node* node : visited){
                if (!node->param) node->d_loss.setZero();
                node->visit = false;
            }
            visited.clear();
        }
};
class ParamNode : public Node{
    public: 
        ParamNode(MatrixXd value, bool requires_grad = true){
            param = true;
            val = value;
            this->requires_grad = requires_grad;
            init_grad_buffer();
        }
        void backward(MatrixXd upstream){
            d_loss += upstream;
        }
};


class AddNode : public Node{
    private:
        Node* xptr;
        Node* yptr;
    public:
        AddNode(Node& x, Node& y){
            xptr = &x;
            yptr = &y;
            if (x.requires_grad) inputs.push_back(&x);
            if (y.requires_grad) inputs.push_back(&y);
            val = x.val + y.val;
            init_grad_buffer();
        }
        void backward(MatrixXd upstream){
            d_loss += upstream;
            walk(this);
            reverse(visited.begin(), visited.end());
            for (Node* node : visited){
                node->propogate();
            }
            reset();
        }
        void propogate(){
            if (xptr->requires_grad) xptr->d_loss += d_loss;
            if (yptr->requires_grad) yptr->d_loss += d_loss;
        }
};

class MultNode : public Node{
    private:
        Node* xptr;
        Node* yptr;
    public:
        MultNode(Node& x, Node& y){
            xptr = &x;
            yptr = &y;
            if (x.requires_grad) inputs.push_back(&x);
            if (y.requires_grad) inputs.push_back(&y);
            /*val = 1;
            for (int i = 0; i<inputs.size();i++){
                val *= inputs[i]->val;
            }*/

            // assume x and y are matrices
            val = x.val * y.val;
            init_grad_buffer();
        }

        void backward(MatrixXd upstream){
            d_loss += upstream;
            walk(this);
            reverse(visited.begin(), visited.end());
            for (Node* node : visited){
                node->propogate();
            }
            reset();
        }
        void propogate(){
            /*for (int i = 0; i<inputs.size();i++){
                grad = 1;
                for (int j = 0; j<inputs.size();j++){
                    if (i!=j){
                        grad *= inputs[j]->val;
                    }
                }
                inputs[i]->d_loss += d_loss*grad;
            }*/
           // Assume y is the column vector
           if (xptr->requires_grad) xptr->d_loss += d_loss * yptr->val.transpose();
           if (yptr->requires_grad) yptr->d_loss += xptr->val.transpose() * d_loss;
        }
};

class LogNode : public Node {
    private:
        Node* xptr;
    public:
        LogNode(Node& x) {
            xptr = &x;
            if (x.requires_grad) inputs.push_back(&x);
            val = x.val.array().log().matrix();
            init_grad_buffer();
        }
        void backward(MatrixXd upstream){
            d_loss+=upstream;
            walk(this);
            reverse(visited.begin(), visited.end());
            for(Node* node : visited){
                node->propogate();
            }
            reset();
        }
        void propogate(){
            if (xptr->requires_grad) xptr->d_loss += d_loss.cwiseProduct(xptr->val.unaryExpr(&LogNode::derivative));
        }
        static double derivative(double val){
            return 1/val;
        }
};

class SigmoidNode : public Node {
    private:
        Node* xptr;
    public:
        static double sigmoid_fcn(double value){
            return 1/(1+exp(0-value));
        }
        SigmoidNode(Node& x){
            xptr = &x;
            if (x.requires_grad) inputs.push_back(&x);
            val = x.val.unaryExpr(&SigmoidNode::sigmoid_fcn);
            init_grad_buffer();
        }
        void backward(MatrixXd upstream){
            d_loss += upstream;
            walk(this);
            reverse(visited.begin(), visited.end());
            for (Node* node : visited){
                node->propogate();
            }
            reset();
        }
        void propogate(){
            if (xptr->requires_grad) xptr->d_loss += d_loss.cwiseProduct(xptr->val.unaryExpr(&SigmoidNode::derivative));
        }
        static double derivative(double value){
            return value*(1-value);
        }
};

class ReLUNode : public Node {
    private:
        Node* xptr;
    public:
        static double relu(double value){
            return (value>0) ? value : 0;
        }
        ReLUNode(Node& x){
            xptr = &x;
            if (x.requires_grad) inputs.push_back(&x);
            val = xptr->val.unaryExpr(&ReLUNode::relu);
            init_grad_buffer();
        }
        void backward(MatrixXd upstream){
            d_loss += upstream;
            walk(this);
            reverse(visited.begin(), visited.end());
            for (Node* node : visited){
                node->propogate();
            }
            reset();
        }
        void propogate(){
            if(xptr->requires_grad) xptr->d_loss += d_loss.cwiseProduct(xptr->val.unaryExpr(&ReLUNode::derivative));
        }
        static double derivative(double value){
            return (value <= 0) ? 0 : 1;
        }
};

class tanhNode : public Node {
    private:
        Node* xptr;
    public:
        tanhNode(Node& x){
            xptr = &x;
            if (x.requires_grad) inputs.push_back(&x);
            val = x.val.array().tanh().matrix();
            init_grad_buffer();
        }
        void backward(MatrixXd upstream){
            d_loss+=upstream;
            walk(this);
            reverse(visited.begin(), visited.end());
            for(Node* node : visited){
                node->propogate();
            }
            reset();
        }
        void propogate(){
            if (xptr->requires_grad) xptr->d_loss += d_loss.cwiseProduct(xptr->val.unaryExpr(&tanhNode::derivative));
        }
        static double derivative(double value){
            return 1-value*value;
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
void zero_params(){
    for (Node* node : global){
        if (node->param) node->d_loss.setZero();
    }
}
void destroy(){
    for (Node* node : global){
        delete node;
    }
    global.clear();
}

#endif