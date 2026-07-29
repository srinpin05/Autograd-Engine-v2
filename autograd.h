#ifndef AUTOGRAD_H
#define AUTOGRAD_H


#include <cmath>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace std;
using namespace Eigen;

// REMEMBER TO CALL zero_params between each backwards() call. 
// ONLY WORKS FOR SINGLE THREADED PROGRAMS.

class Node {
    public:
        vector<Node*> inputs;
        vector<Node*> visited;
        bool heap_owned = false;
        bool param = false;
        Tensor<double, 3> d_loss;
        int forward_count = 0;
        int backward_count = 0;
        bool visit = false;
        bool requires_grad = true; 
        Tensor<double, 3> val;
        virtual void backward(MatrixXd upstream){
            d_loss += upstream;
            walk(this);
            reverse(visited.begin(), visited.end());
            for (Node* node : visited){
                node->propogate();
            }
            reset();
        }
        virtual void propogate(){} //helper function to push local gradient down to inputs.
        virtual ~Node(){}
        void init_grad_buffer(){
            d_loss.resize(val.dimensions());
            d_loss.setZero();
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

vector<Node*> global;
class ParamNode : public Node{
    public: 
        Tensor<double, 2> val;
        ParamNode(Tensor<double, 2> value, bool requires_grad = true, bool heap_owned = false){
            param = true;
            val = value;
            this->requires_grad = requires_grad;
            this->heap_owned = heap_owned;
            init_grad_buffer();
            global.push_back(this);
        }
        void backward(MatrixXd upstream){
            d_loss += upstream;
        }
};

class InputNode : public Node{
    public: 
        InputNode(Tensor<double, 3> value, bool requires_grad = true, bool heap_owned = false){
            val = value;
            this->requires_grad = requires_grad;
            this->heap_owned = heap_owned;
            init_grad_buffer();
            global.push_back(this);
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
            global.push_back(this);
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
            Eigen::array<Eigen::IndexPair<int>,1> product_dims = {
                Eigen::IndexPair<int>(1,1)
            };
            //val = x.val * y.val;
            val = y.val.contract(x.val, product_dims).shuffle(Eigen::array<int, 3>{0,2,1});
            init_grad_buffer();
            global.push_back(this);
        }
        void propogate(){
           // Assume y is the column vector
            Eigen::array<Eigen::IndexPair<int>,2> x_dims = {
                Eigen::IndexPair<int>(0,0), Eigen::IndexPair<int>(2,2)
            };

           //if (xptr->requires_grad) xptr->d_loss += d_loss * yptr->val.transpose();
           if (xptr->requires_grad) {
                Tensor<double, 2> grad_x = d_loss.contract(yptr->val, x_dims);
                xptr->d_loss += grad_x * (1.0 / 32);

           }
           //if (yptr->requires_grad) yptr->d_loss += xptr->val.transpose() * d_loss;
           Eigen::array<Eigen::IndexPair<int>, 1> y_dims = {
                IndexPair<int>(1,0)
           };
           if (yptr->requires_grad){
                Tensor<double, 3> grad_y = d_loss.contract(xptr->val, y_dims);
                yptr->d_loss += grad_y.shuffle(Eigen::array<int,3>{0,2,1});
           }

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
            global.push_back(this);
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
            global.push_back(this);
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
            global.push_back(this);
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
            global.push_back(this);
        }
        void propogate(){
            if (xptr->requires_grad) xptr->d_loss += d_loss.cwiseProduct(xptr->val.unaryExpr(&tanhNode::derivative));
        }
        static double derivative(double value){
            return 1-value*value;
        }
};

//LOSS FUNCTION NODES


class MSENode : public Node {
    private:
        Node* xptr;
        Node* yptr;
        MatrixXd error;
        double N;
    public: 
        //X is predicted and Y is true value
        MSENode(Node& x, Node& y){
            xptr = &x;
            yptr = &y;
            if (x.requires_grad) inputs.push_back(&x);
            N = x.val.rows();
            error = x.val - y.val;
            val = MatrixXd::Constant(1, 1, (1/N)*((error.cwiseProduct(error)).sum()));
            init_grad_buffer();
            global.push_back(this);
        }
        void propogate(){
            if (xptr->requires_grad) xptr->d_loss += (2*error)/N;
        }
};

inline Node& operator*(Node& x, Node& y){
    Node* mult = new MultNode(x, y);
    mult->heap_owned = true;
    return *mult;
}
inline Node& operator+(Node& x, Node& y){
    Node* add = new AddNode(x, y);
    add->heap_owned = true;
    return *add;
}
inline void zero_params(){
    for (Node* node : global){
        if (node->param) node->d_loss.setZero();
    }
}
inline void print_gradient_params(){
    for (Node* node : global){
        if (node->param) cout<<node->d_loss<<endl;
    }
}
inline void update_gradient_params(double learning_rate){
    for (Node* node : global){
        if (node->param) node->val -= learning_rate*node->d_loss;
    }
}
inline void destroy(){
    for (Node* node : global){
        if (node->heap_owned) delete node;
    }
    global.clear();
}

#endif