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
/* 
Three different approaches considered for Batched autograd
 1. Loop over batches and epochs seperately in network.h and leave autograd.h 
    (slow but easy)

 2. Assume all tensors that are passed into autograd.h are 3 dimensional (for batches)
    and treat the 0th dimension as iteration count
    and operate on the other 2 dimensions as 2d matrices 
    (mid but nice generalization)

 3. Write specific code for each Node with specific assumptions 
    about the 2d matrices (parameters) and the 3d matrices (batched data) 
    
    (faster but no generalization)
*/


//FORWARD DECLARATIONS

inline double sigmoid_fcn(double value);
inline double sigmoid_derivative(double value);
inline double relu(double value);
inline double relu_derivative(double value);
inline double tanh_derivative(double value);
inline void softmax(const Tensor<double, 3> &input, Tensor<double, 3> &val);
inline Tensor<double, 3> softmax(const Tensor<double, 3> &input);
inline Tensor<double, 3> clamp(Tensor<double, 3> input);
//NODE CLASSES

class Node {
    public:
        vector<Node*> inputs;
        vector<Node*> visited;
        bool input = false;
        bool heap_owned = false;
        bool param = false;
        Tensor<double, 3> d_loss;
        int forward_count = 0;
        int backward_count = 0;
        bool visit = false;
        bool requires_grad = true; 
        Tensor<double, 3> val;
        virtual void backward(Tensor<double,3> upstream){
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
        ParamNode(Tensor<double, 3> value, bool requires_grad = true){
            param = true;
            val = value;
            this->requires_grad = requires_grad;
            this->heap_owned = true;
            init_grad_buffer();
            global.push_back(this);
        }
        void backward(Tensor<double,3> upstream){
            d_loss += upstream;
        }
};

class InputNode : public Node{
    public: 
        InputNode(Tensor<double, 3> value, bool requires_grad = true){
            input = true;
            val = value;
            this->requires_grad = requires_grad;
            this->heap_owned = true;
            init_grad_buffer();
            global.push_back(this);
        }
        void backward(Tensor<double,3> upstream){
            d_loss += upstream;
        }
};



//BASIC OPERATOR NODES

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

            if (x.val.dimension(0) == y.val.dimension(0)) {
                // batch dims already match (or both are size 1) — no loop needed
                val = x.val + y.val;
            } else {
                // batch dims differ (one side is a broadcastable size-1 operand) — loop over batch
                int batch = std::max(x.val.dimension(0), y.val.dimension(0));
                val = Tensor<double,3>(batch, x.val.dimension(1), x.val.dimension(2));
                for (int b = 0; b < batch; b++){
                    int xb = (x.val.dimension(0) == 1) ? 0 : b;
                    int yb = (y.val.dimension(0) == 1) ? 0 : b;
                    Tensor<double,2> x_slice = x.val.chip(xb, 0);
                    Tensor<double,2> y_slice = y.val.chip(yb, 0);
                    val.chip(b, 0) = x_slice + y_slice;
                }
            }
            init_grad_buffer();
            global.push_back(this);
        }
        void propogate(){
            int batch = d_loss.dimension(0);

            if (xptr->requires_grad) {
                if (xptr->val.dimension(0) == batch) {
                    xptr->d_loss += d_loss;
                } else {
                    // xptr was broadcast (batch=1) — sum contributions over batch
                    for (int b = 0; b < batch; b++){
                        Tensor<double,2> d_loss_slice = d_loss.chip(b, 0);
                        xptr->d_loss.chip(0, 0) += d_loss_slice;
                    }
                }
            }
            if (yptr->requires_grad) {
                if (yptr->val.dimension(0) == batch) {
                    yptr->d_loss += d_loss;
                } else {
                    for (int b = 0; b < batch; b++){
                        Tensor<double,2> d_loss_slice = d_loss.chip(b, 0);
                        yptr->d_loss.chip(0, 0) += d_loss_slice;
                    }
                }
            }
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

            int batch = std::max(x.val.dimension(0), y.val.dimension(0));
            int out_rows = x.val.dimension(1);
            int out_cols = y.val.dimension(2);

            val = Tensor<double,3>(batch, out_rows, out_cols);

            Eigen::array<Eigen::IndexPair<int>,1> product_dims_2d = {
                Eigen::IndexPair<int>(1,0) // x_slice's cols against y_slice's rows
            };

            for (int b = 0; b < batch; b++){
                int xb = (x.val.dimension(0) == 1) ? 0 : b;
                int yb = (y.val.dimension(0) == 1) ? 0 : b;
                Tensor<double,2> x_slice = x.val.chip(xb, 0);
                Tensor<double,2> y_slice = y.val.chip(yb, 0);
                val.chip(b, 0) = x_slice.contract(y_slice, product_dims_2d);
            }

            init_grad_buffer();
            global.push_back(this);
        }

        void propogate(){
            int batch = d_loss.dimension(0);

            Eigen::array<Eigen::IndexPair<int>,1> dx_dims = { Eigen::IndexPair<int>(1,1) };
            Eigen::array<Eigen::IndexPair<int>,1> dy_dims = { Eigen::IndexPair<int>(0,0) };

            for (int b = 0; b < batch; b++){
                Tensor<double,2> d_loss_slice = d_loss.chip(b, 0);

                if (xptr->requires_grad) {
                    int yb = (yptr->val.dimension(0) == 1) ? 0 : b;
                    Tensor<double,2> y_slice = yptr->val.chip(yb, 0);
                    Tensor<double,2> grad_x_slice = d_loss_slice.contract(y_slice, dx_dims);
                    int xb = (xptr->val.dimension(0) == 1) ? 0 : b;
                    xptr->d_loss.chip(xb, 0) += grad_x_slice;
                }

                if (yptr->requires_grad) {
                    int xb = (xptr->val.dimension(0) == 1) ? 0 : b;
                    Tensor<double,2> x_slice = xptr->val.chip(xb, 0);
                    Tensor<double,2> grad_y_slice = x_slice.contract(d_loss_slice, dy_dims);
                    int yb = (yptr->val.dimension(0) == 1) ? 0 : b;
                    yptr->d_loss.chip(yb, 0) += grad_y_slice;
                }
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
            val = x.val.log(); // elementwise across all 3 axes at once — no loop needed
            init_grad_buffer();
            global.push_back(this);
        }
        void propogate(){
            if (xptr->requires_grad) xptr->d_loss += d_loss * xptr->val.unaryExpr(&LogNode::derivative);
        }
        static double derivative(double val){
            return 1/val;
        }
};



//COMMON ACTIVATION FUNCTION NODES

class SigmoidNode : public Node {
    private:
        Node* xptr;
    public:
        SigmoidNode(Node& x){
            xptr = &x;
            if (x.requires_grad) inputs.push_back(&x);
            val = x.val.unaryExpr(&sigmoid_fcn); // elementwise — no loop needed
            init_grad_buffer();
            global.push_back(this);
        }
        void propogate(){
            if (xptr->requires_grad) xptr->d_loss += d_loss * xptr->val.unaryExpr(&sigmoid_derivative);
        }
};

class ReLUNode : public Node {
    private:
        Node* xptr;
    public:
        ReLUNode(Node& x){
            xptr = &x;
            if (x.requires_grad) inputs.push_back(&x);
            val = xptr->val.unaryExpr(&relu); // elementwise — no loop needed
            init_grad_buffer();
            global.push_back(this);
        }
        void propogate(){
            if(xptr->requires_grad) xptr->d_loss += d_loss * xptr->val.unaryExpr(&relu_derivative);
        }
};

class tanhNode : public Node {
    private:
        Node* xptr;
    public:
        tanhNode(Node& x){
            xptr = &x;
            if (x.requires_grad) inputs.push_back(&x);
            val = x.val.tanh(); // elementwise — no loop needed
            init_grad_buffer();
            global.push_back(this);
        }
        void propogate(){
            if (xptr->requires_grad) xptr->d_loss += d_loss * xptr->val.unaryExpr(&tanh_derivative);
        }
};

class SoftmaxNode : public Node {
    private:
        Node *xptr;
        int batch;
        int logits;
    public:
        SoftmaxNode(Node& x){
            //Shape has to be (b, m, 1) where b is batch and m is the number of logits. 
            xptr = &x;
            if (x.requires_grad) inputs.push_back(&x);
            batch = x.val.dimension(0);
            logits = x.val.dimension(1);
            val = Tensor<double, 3>(batch, logits, 1);
            softmax(x.val, val);
            init_grad_buffer();
            global.push_back(this);
        } 
        void propogate(){
            if (xptr->requires_grad) {
                for (int b = 0; b < batch; b++){
                    Tensor<double, 2> temp = val.chip(b,0);
                    for (int i = 0; i < logits; i++){
                        double Si = temp(i, 0);
                        double sum = 0;
                        for (int j = 0; j<logits; j++){
                            double Sj = temp(j, 0);
                            sum += ((i!=j) ? (0 - Si*Sj) : Si*(1-Si)) * d_loss(b, j, 0);
                        }
                        xptr->d_loss(b, i, 0) += sum;
                    }
                }
            }
        }
};



//LOSS FUNCTION NODES

class MSENode : public Node {
    private:
        Node* xptr;
        Node* yptr;
        Tensor<double,3> error;
        double N;
    public: 
        //X is predicted and Y is true value
        MSENode(Node& x, Node& y){
            xptr = &x;
            yptr = &y;
            if (x.requires_grad) inputs.push_back(&x);
            N = x.val.dimension(0) * x.val.dimension(1); // batch size, 0th dimension
            error = x.val - y.val;  // elementwise across all 3 axes — no loop needed
            Eigen::Tensor<double,0> sum_sq = (error * error).sum();
            val = Tensor<double,3>(1,1,1);
            val(0,0,0) = (1.0/N) * sum_sq(0);
            init_grad_buffer();
            global.push_back(this);
        }
        void propogate(){
            if (xptr->requires_grad) xptr->d_loss += ((2*error)/N) * d_loss(0,0,0);
        }
};


class CategoricalCrossEntropyNode : public Node {
    private:
        Node* xptr;
        Node* yptr;
        double N;
        Tensor<double, 3> L;
        Tensor<double, 3> error;
        Tensor<double,3> x_clamped;
    public: 
        //X is predicted and Y is true value
        CategoricalCrossEntropyNode(Node& x, Node& y){
            xptr = &x;
            yptr = &y;
            if (x.requires_grad) inputs.push_back(&x);

            //Clamping to keep log defined
            x_clamped = clamp(x.val);

            N = x.val.dimension(0); // batch size, 0th dimension
            error = x.val - y.val;

            //Average across all batches to get scalar (no intermediate L tensor)
            Eigen::Tensor<double,0> sum_across_batches = ((y.val * (x_clamped).log()).sum(std::array<int, 1> {1})).sum();
            val = Tensor<double,3>(1,1,1);
            val(0,0,0) = -(1.0/N) * sum_across_batches(0);

            init_grad_buffer();
            global.push_back(this);
        }
        void propogate(){
            if (xptr->requires_grad) xptr->d_loss +=  (-(yptr->val/x_clamped))/N * d_loss(0,0,0);
        }
};


class CrossEntropyNode : public Node {
    private:
        Node* xptr;
        Node* yptr;
        double N;
        Tensor<double, 3> L;
        Tensor<double, 3> error;
        Tensor<double,3> x_clamped;
    public: 
        //X is predicted and Y is true value
        CrossEntropyNode(Node& x, Node& y){
            xptr = &x;
            yptr = &y;
            if (x.requires_grad) inputs.push_back(&x);

            //Clamping to keep log defined
            x_clamped = clamp(x.val);

            N = x.val.dimension(0); // batch size, 0th dimension
            error = x.val - y.val;

            //Average across all batches to get scalar (no intermediate L tensor)
            Eigen::Tensor<double,0> sum_across_batches = (-(y.val * (x_clamped).log()) + (-y.val + 1.0) * (-x_clamped + 1.0).log()).sum();
            val = Tensor<double,3>(1,1,1);
            val(0,0,0) = (1.0/N) * sum_across_batches(0);

            init_grad_buffer();
            global.push_back(this);
        }
        void propogate(){
            if (xptr->requires_grad) xptr->d_loss += (((error) / ((x_clamped)*(-x_clamped + 1.0)))/N) * d_loss(0,0,0);
        }
};


class SigmoidBCENode : public Node {
    private:
        Node* xptr;
        Node* yptr;
        double N;
        Tensor<double, 3> L;
        Tensor<double, 3> error;
        Tensor<double,3> x_sigmoid;
    public:
        //X is predicted and Y is true value
        SigmoidBCENode(Node& x, Node& y){
            xptr = &x;
            yptr = &y;
            if (x.requires_grad) inputs.push_back(&x);

            //Clamping to keep log defined
            N = x.val.dimension(0);

            //Clamped sigmoid
            x_sigmoid = clamp(x.val.unaryExpr(&sigmoid_fcn));
            error = x_sigmoid - y.val;

            //Average across all batches to get scalar (no intermediate L tensor)
            Eigen::Tensor<double,0> sum_across_batches = (-(y.val * (x_sigmoid).log() + (-y.val + 1.0) * (-x_sigmoid + 1.0).log())).sum();
            val = Tensor<double,3>(1,1,1);
            val(0,0,0) = (1.0/N) * sum_across_batches(0);

            init_grad_buffer();
            global.push_back(this);
        }
        void propogate(){
            if (xptr->requires_grad) xptr->d_loss += (error/N) * d_loss(0,0,0);
        }
};



class SoftmaxCCENode : public Node {
    private:
        Node* xptr;
        Node* yptr;
        double N;
        Tensor<double, 3> L;
        Tensor<double, 3> error;
        Tensor<double,3> x_softmax;
    public:
        //X is predicted and Y is true value
        SoftmaxCCENode(Node& x, Node& y){
            xptr = &x;
            yptr = &y;
            if (x.requires_grad) inputs.push_back(&x);

            //Clamping to keep log defined
            N = x.val.dimension(0);

            //Clamped softmax
            softmax(x.val, x_softmax);
            x_softmax = clamp(x_softmax);
            error = x_softmax - y.val;

            //Average across all batches to get scalar (no intermediate L tensor)
            Eigen::Tensor<double,0> sum_across_batches = ((y.val * (x_softmax).log()).sum(std::array<int, 1> {1})).sum();
            val = Tensor<double,3>(1,1,1);
            val(0,0,0) = -(1.0/N) * sum_across_batches(0);

            init_grad_buffer();
            global.push_back(this);
        }
        void propogate(){
            if (xptr->requires_grad) xptr->d_loss += (error/N) * d_loss(0,0,0);
        }
};



//ACTIVATION FUNCTIONS AND THEIR DERIVATIVES

inline double sigmoid_fcn(double value){
    return 1/(1+exp(0-value));
}
inline double sigmoid_derivative(double value){
    return value*(1-value);
}
inline double relu(double value){
    return (value>0) ? value : 0;
}
inline double relu_derivative(double value){
    return (value <= 0) ? 0 : 1;
}
inline double tanh_derivative(double value){
    return 1-value*value;
}


//OPERATOR OVERLOADERS

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

//HELPER FUNCTIONS

inline Tensor<double, 3> clamp(Tensor<double, 3> input){
    double eps = 1e-7;
    return input.cwiseMax(input.constant(eps)).cwiseMin(input.constant(1.0-eps));
            
}
inline void softmax(const Tensor<double, 3> &input, Tensor<double, 3> &val){
    double batch = input.dimension(0);
    double logits = input.dimension(1);
    val = Tensor<double, 3>(batch, logits, 1);

    for (int i = 0; i < batch; i++) {
        Eigen::Tensor<double, 2> row = input.chip(i, 0);
        Eigen::Tensor<double, 1> row_max = row.maximum(std::array<int, 1>{0});
        Eigen::Tensor<double, 1> row_sum = (row - row_max(0)).exp().sum(std::array<int, 1>{0});
        val.chip(i, 0) = (row - row_max(0)).exp() / row_sum(0);
    }
}

inline Tensor<double, 3> softmax(const Tensor<double, 3> &input){
    double batch = input.dimension(0);
    double logits = input.dimension(1);
    Tensor<double, 3> val(batch, logits, 1);

    for (int i = 0; i < batch; i++) {
        Eigen::Tensor<double, 2> row = input.chip(i, 0);
        Eigen::Tensor<double, 1> row_max = row.maximum(std::array<int, 1>{0});
        Eigen::Tensor<double, 1> row_sum = (row - row_max(0)).exp().sum(std::array<int, 1>{0});
        val.chip(i, 0) = (row - row_max(0)).exp() / row_sum(0);
    }
    return val;
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
inline void destroy_mid(){
    vector<Node*> keep;
    keep.reserve(global.size());
    for (Node* node : global){
        if (node->param || node->input) keep.push_back(node);
        else delete node;
    }
    global = std::move(keep);
}

inline void remove_from_global(Node* target){
    global.erase(std::remove(global.begin(), global.end(), target), global.end());
}

#endif