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

void Network::print(){
    std::cout << "Printing network layout..." << std::endl;
    for(unsigned i = 1; i< layers.size(); i++)
        layers[i].print();
};

void Network::train(const std::vector<Sample> &data, const std::vector<double> target_values, const unsigned epochs){
    std::cout << data.size() << " : " << target_values.size() << std::endl;
    assert(data.size() == target_values.size());
    std::vector<double> output_targets;
    for(unsigned i = 0; i < epochs; i++){
        std::cout << "Beginning Epoch " << i << std::endl;
        for(unsigned z = 0; z < data.size(); z++){
            std::cout << "Sample " << z << std::endl;
            // start forward pass
            forwards(data[z]);
            // create array of 0.0 values, expect for the target value of the sample, which will be 1.0
            for(unsigned y = 0; y < data[z].size(); y++)
                output_targets.push_back(y == target_values[z] ? 1.0 : 0.0);

            backwards(output_targets);
            output_targets.clear();
            print();
        }
    }
};

void Network::forwards(const std::vector<double> &data){
    // Assing input data values to input layer for processing
    std::vector<Neuron> &input_layer = layers[0].getNeurons();
    for(unsigned i = 0; i < data.size(); i++)
        input_layer[i].setOutput(data[i]);
    
    // Begin forward propagation logic
    std::cout << "Beginning foward propagation.." << std::endl;
    //start at layer_num = 1 to skip input layer, since it does nothing other than hold input values
    for(unsigned layer_num = 1; layer_num < layers.size(); layer_num++){
        std::cout << "Layer " << layer_num << std::endl;
        std::vector<Neuron> &prev_layer = layers[layer_num - 1].getNeurons();
        std::vector<Neuron> &curr_layer = layers[layer_num].getNeurons();
        for(unsigned i = 0; i < curr_layer.size(); i++)
            curr_layer[i].forwards(prev_layer);
    }
    std::cout << "Completed forward propagation." << std::endl;
};

void Network::backwards(const std::vector<double> &target_values){
    std::cout << "Beginning backwards propagation..." << std::endl;
    std::vector<Neuron> &output_layer = layers.back().getNeurons();
    error = 0.0;
    // Calculate output error for processed sample.
    for(unsigned i = 0; i < output_layer.size(); i++){
        double delta = target_values[i] - output_layer[i].getOutput();
        error += delta * delta;
    }

    error /= output_layer.size();
    error = sqrt(error);
    std::cout << "Error: " << error << std::endl;

    // calculate the output layer gradient
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

    // recursively update weights for all layers, (except input layer)
    for(unsigned layer_num = layers.size() - 1; layer_num > 0; layer_num--){
        std::vector<Neuron> &curr_layer = layers[layer_num].getNeurons();
        std::vector<Neuron> &prev_layer = layers[layer_num - 1].getNeurons();
        for(unsigned i = 0; i < curr_layer.size(); i++)
            curr_layer[i].updateInputWeights(prev_layer);
    }

    std::cout << "Completed backpropagation." << std::endl;
};