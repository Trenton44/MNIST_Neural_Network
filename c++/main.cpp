#include <vector>
#include <iostream>
#include <fstream>

#include "csv.cpp"
#include "neuron.cpp"
#include "layer.cpp"
#include "network.cpp"
typedef std::vector<double> Sample;
void printVector(std::vector<Sample> &data){
    std::cout << "printing sample data." << std::endl;
    for(unsigned i = 0; i < data.size(); i++) {
        for(unsigned z = 0; z < data[i].size(); z++)
            std::cout << data[i][z] << ", ";
    }
    std::cout << std::endl;
}
void printVector(std::vector<double> &data){
    std::cout << "printing vector<double> (length " << data.size() << "): " << std::endl;
    for(unsigned i = 0; i < data.size(); i++)
        std::cout << data[i] << ", ";
    std::cout << std::endl;
}

int main(){
    std::vector<unsigned> topology;
    topology.push_back(28 * 28); // input layer, will not be modified apart from setting the output values to match the input data
    topology.push_back(16);
    topology.push_back(16);
    topology.push_back(10);

    Network net(topology);

    std::vector<Sample> train_data;
    std::vector<Sample> test_data;
    std::vector<double> target_train_values;
    std::vector<double> target_test_values;

    // load and parse train and test data
    readCSV(train_data, "../mnist_train.csv");
    parseDataset(train_data, target_train_values);
    readCSV(test_data, "../mnist_test.csv");
    parseDataset(test_data, target_test_values);

    net.train(train_data, target_train_values, 35);
}