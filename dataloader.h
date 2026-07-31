#include<fstream>
#include<iostream>
#include<vector>
#include "autograd.h"
#include<Eigen/Dense>

//MNIST is in big-endian format
using namespace std;

int size_of_sample;
int total_size;
int reverseInteger(int i){
    //c1 is LSByte and c4 is MSByte
    unsigned char c1 = i & 255;
    unsigned char c2 = (i >> 8) & 255;
    unsigned char c3 = (i >> 16) & 255;
    unsigned char c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + (int)c4;
}

void loadData8bit(const string &path, vector<uint8_t>& datas){
    ifstream file(path, ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open image file: " << path << std::endl;
        return;
    }

    int magic_number = 0, num_images = 0, rows = 0, cols = 0;
    
    // Read Metadata headers
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_images, sizeof(num_images));
    file.read((char*)&rows, sizeof(rows));
    file.read((char*)&cols, sizeof(cols));

    magic_number = reverseInteger(magic_number);
    num_images = reverseInteger(num_images);

    if (magic_number != 2051) { 
        std::cerr << "Invalid MNIST label file magic number!" << std::endl;
        return;
    }

    rows = reverseInteger(rows);
    cols = reverseInteger(cols);
    
    //Dynamic Allocation
    size_t data_size = (size_t)num_images * rows * cols;
    size_of_sample = rows*cols;
    total_size = num_images;
    datas.resize(data_size);

    //Read Data
    file.read((char*)datas.data(), num_images*rows*cols);
}

void readLabels8bit(const std::string& path, std::vector<uint8_t>& labels) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open label file: " << path << std::endl;
        return;
    }

    int magic_number = 0, num_items = 0;
    
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_items, sizeof(num_items));

    magic_number = reverseInteger(magic_number);
    num_items = reverseInteger(num_items);

    if (magic_number != 2049) { 
        std::cerr << "Invalid MNIST label file magic number!" << std::endl;
        return;
    }

    labels.resize(num_items);
    file.read((char*)labels.data(), num_items);
}

void dataToTensor(vector<uint8_t> &datas, Tensor<double, 1> &input_data){
    TensorMap<Tensor<uint8_t, 1>> map(datas.data(), datas.size());
    input_data = map.cast<double>();
}
void labelToTensor(vector<uint8_t> &datas, Tensor<double, 1> &input_data){
    TensorMap<Tensor<uint8_t, 1>> map(datas.data(), datas.size());
    input_data = map.cast<double>();
}
void OneHotEncoder(Tensor<double, 1> &input_data, Tensor<double, 2> &target, int num_classes){
    int total = input_data.dimension(0);
    Tensor<double, 2> a(total, num_classes);
    a.setZero();
    for (int i = 0; i < total; i++){
        a(i, (int)input_data(i)) = 1;
    }
    target = a;

}

inline void LoadData(Tensor<double, 1> &x, Tensor<double, 2>& y_encoded, const string &path1, const string &path2, int num_classes, bool one_hot = true){
    vector<uint8_t> Xs; 
    vector<uint8_t> Ys;
    Tensor<double, 1> y;
    loadData8bit(path1, Xs);
    readLabels8bit(path2, Ys);
    dataToTensor(Xs, x);
    labelToTensor(Ys, y);
    OneHotEncoder(y, y_encoded, num_classes);
}