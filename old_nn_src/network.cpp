#include <vector>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <iostream>

#include "network.hpp"
#include "neuron.hpp"
#include "layer.hpp"

Network::Network(const std::vector<unsigned> &topology){
    // create each layer, passing in id, #neurons, & #weights connecting to the next layer.
    // if last layer, 0 weights (there's no layer in front for it to hold weights for.)
    for(unsigned layer_num = 0; layer_num < topology.size(); layer_num++)
        layers.push_back(Layer(layer_num, topology[layer_num], layer_num == topology.size() - 1 ? 0 : topology[layer_num + 1]));
};

void Network::predict(const std::vector<double> &data){ forwards(data); };

double Network::train(std::string network_save_filename, const std::vector<Sample> &data, const std::vector<double> target_values, const unsigned epochs, const unsigned sample_count = 0){
    assert(data.size() == target_values.size());
    std::vector<double> output_targets;
    double error_low = 100;
    double error_current;
    unsigned samples_length = sample_count == 0 ? data.size() : sample_count;
    for(unsigned i = 0; i < epochs; i++){
        std::cout << "beginning Epoch " << i << std::endl;
        // std::cout << "Samples to be run: " << samples_length << std::endl;
        for(unsigned z = 0; z < samples_length; z++){
            // std::cout << "Sample " << z << std::endl;
            // start forward pass
            forwards(data[z]);
            // create array of 0.0 values, except for th target value of the sample, which will be 1.0
            for(unsigned y = 0; y < data[z].size(); y++)
                output_targets.push_back(y == target_values[z] ? 1.0 : 0.0);
            
            error_current = backwards(output_targets);
            output_targets.clear();
            if(error_low - error_current > 0.001){
                std::cout << "Epoch " << i << ":" << z << " resulted in " << std::scientific << error_current << ". Saving new best to file. " << std::endl;
                save(network_save_filename);
                error_low = error_current;
            }
            // print()
        }
    }
    return error_low;
};

void Network::forwards(const std::vector<double> &data){
    // Assing input data values to input layer for processing
    std::vector<Neuron> &input_layer = layers[0].getNeurons();
    for(unsigned i = 0; i < data.size(); i++)
        input_layer[i].setOutput(data[i]);
    
    // Begin forward propagation logic
    // std::cout << "Beginning foward propagation.." << std::endl;
    // start at layer_num = 1 to skip input layer, since it does nothing other than hold input values
    for(unsigned layer_num = 1; layer_num < layers.size(); layer_num++){
        //std::cout << "Layer " << layer_num << std::endl;
        std::vector<Neuron> &prev_layer = layers[layer_num - 1].getNeurons();
        std::vector<Neuron> &curr_layer = layers[layer_num].getNeurons();
        for(unsigned i = 0; i < curr_layer.size(); i++)
            curr_layer[i].forwards(prev_layer);
    }
    // std::cout << "Completed forward propagation." << std::endl;
};

double Network::backwards(const std::vector<double> &target_values){
    //std::cout << "Beginning backwards propagation..." << std::endl;
    std::vector<Neuron> &output_layer = layers.back().getNeurons();
    error = 0.0;
    // Calculate output error for processed sample.
    // Cost function of node is Mean Squared Error: (expected - actual)^2, which is calculated here
    for(unsigned i = 0; i < output_layer.size(); i++){
        double delta = target_values[i] - output_layer[i].getOutput(); // (expected - actual)
        error += delta * delta; // ^2
    }

    // Cost of the network is the average cost of all output nodes
    error /= output_layer.size();
    error = sqrt(error);
    // std::cout << "Error: " << error << std::endl;

    // calculate the output layer gradient
    // change in cost relative to a node is 
    for(unsigned i = 0; i < output_layer.size(); i++)
        output_layer[i].getOutputGradients(target_values[i]);
    
    // calculate gradients on hidden layers.
    // calculate backwards from output layer. layer_num starts at size - 2 to skip output layer, and stops before 0 to skip input layer.
    for(unsigned layer_num = layers.size() - 2; layer_num > 0; layer_num--){
        std::vector<Neuron> &hidden_layer = layers[layer_num].getNeurons();
        std::vector<Neuron> &next_layer = layers[layer_num + 1].getNeurons();
        for(unsigned i = 0; i < hidden_layer.size(); i++)
            hidden_layer[i].getHiddenGradients(next_layer);
    }

    // recursively update weights/bias for all layers, (except input layer, since it's not really a layer.)
    for(unsigned layer_num = layers.size() - 1; layer_num > 0; layer_num--){
        std::vector<Neuron> &curr_layer = layers[layer_num].getNeurons();
        std::vector<Neuron> &prev_layer = layers[layer_num - 1].getNeurons();
        for(unsigned i = 0; i < curr_layer.size(); i++)
            curr_layer[i].updateInputWeights(prev_layer);
    }

    // std::cout << "Completed backpropagation." << std::endl;
    return error;
};

bool Network::save(std::string filename){
    // first row of csv = topology
    // all remaining rows contain neuron data

    std::cout << "Opening " << filename << std::endl;
    std::fstream file;
    file.open(filename, std::fstream::out);
    std::cout << "Successfully opened " << filename << std::endl;
    for(unsigned i = 0; i < layers.size(); i++){
        file << layers[i].getNeurons().size();
        if(i < layers.size() - 1)
            file << ",";
    }
    for(unsigned i = 0; i < layers.size(); i++){ 
        std::vector<Neuron> &curr_layer = layers[i].getNeurons();
        for(unsigned z = 0; z < curr_layer.size(); z++){
            file << std::endl;
            curr_layer[z].save(file); 
        }
    }
    std::cout << "Network data successfully saved to " << filename << std::endl;
    file.close();
    return true;
};

Network Network::load(std::string filename){
    std::cout << "Opening " << filename << std::endl;
    std::fstream file;
    std::string str;
    std::vector<std::vector<double>> network_data;
    std::vector<double> line_data;
    std::string tempstr;

    file.open(filename, std::fstream::in);
    if(file.good())
        std::cout << "Successfully opened " << filename << std::endl;
    else{
        std::cout << "Unable to open file." << std::endl;
        throw;
    }
    while(std::getline(file, str)){
        for(unsigned cursor = 0; cursor < str.length(); cursor++){
            if(str[cursor] == ','){
                line_data.push_back(std::stod(tempstr));
                tempstr.clear();
            }
            else
                tempstr.push_back(str[cursor]);
        }
        // add last value that didn't have a following comma, to data.
        line_data.push_back(std::stod(tempstr));

        network_data.push_back(line_data);
        tempstr.clear();
        line_data.clear();
        str.clear();
    }
    file.close();
    std::cout << "Successfully read file into memory, parsing data..." << std::endl;

    // first line of data should contain topology
    std::vector<unsigned> topology(network_data[0].begin(), network_data[0].end());
    Network net(topology);
    unsigned layer_num = 0;
    unsigned cursor = 0;
    for(unsigned sample = 1; sample < network_data.size(); sample++){
        if(cursor >= net.layers[layer_num].getNeurons().size()){
            layer_num += 1;
            cursor = 0;
        }
        std::vector<Neuron> &current_layer = net.layers[layer_num].getNeurons();
        current_layer[cursor].load(network_data[sample]);
        cursor += 1;
    }
    return net;
};

unsigned Network::results(void){
    std::vector<Neuron> &output_layer = layers.back().getNeurons();
    unsigned guess = 0;
    double max = output_layer[0].getOutput();
    for(unsigned i = 1; i < output_layer.size(); i++){
        if(output_layer[i].getOutput() > max){
            max = output_layer[i].getOutput();
            guess = i;
        }
    }
    return guess;
};

void Network::print(){
    std::cout << "Printing network layout..." << std::endl;
    for(unsigned i = 1; i< layers.size(); i++)
        layers[i].print();
};