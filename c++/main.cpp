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

unsigned test(std::string network_save_filename, std::string test_data_filename){
    std::vector<Sample> test_data;
    std::vector<double> target_test_values;
    readCSV(test_data, test_data_filename, 1.0);
    parseDataset(test_data, target_test_values);
    
    Network net = Network::load(network_save_filename);
    double t_error = 0;
    unsigned result;
    for(unsigned i = 0; i < test_data.size(); i++){
        net.predict(test_data[i]);
        result = net.results();
        if(result != target_test_values[i])
            t_error += 1;
    }
    std::cout << "Final result: " << (t_error / test_data.size()) * 100 << std::endl;
    return t_error;
}

double train(std::string network_save_filename, std::string train_data_filename){
    std::vector<unsigned> topology;
    std::vector<Sample> train_data;
    std::vector<double> target_train_values;
    
    topology.push_back(28 * 28); // input layer, will not be modified apart from setting the output values to match the input data
    topology.push_back(16);
    topology.push_back(16);
    topology.push_back(10);

    Network net(topology);

    readCSV(train_data, train_data_filename, 255.0);
    parseDataset(train_data, target_train_values);

    double error = net.train(network_save_filename, train_data, target_train_values, 15);
    std::cout << "Network training completed. Lowest error: " << error << std::endl;
    std::cout << "Network state resulting in this value has ben saved to " << network_save_filename << std::endl;
    return error;
}
int main(){
    std::string network_save_filename = "network_save.csv";
    std::string test_data_filename = "../mnist_test.csv";
    std::string train_data_filename = "../mnist_train.csv";
    //train(network_save_filename, train_data_filename);
    test(network_save_filename, test_data_filename);
}