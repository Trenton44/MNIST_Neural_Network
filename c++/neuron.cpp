#include <vector>
#include <iostream>
#include <cmath>
#include "layer.hpp"
#include "neuron.hpp"

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

Neuron::Neuron(unsigned id, unsigned outputs){
    output = 0;
    gradient = 0;
    neuron_id = id;
    for(unsigned i = 0; i < outputs; i++){
        weights.push_back(Connection());
        weights.back().weight = generateWeight();
    }
}

void Neuron::forwards(std::vector<Neuron> &prev_layer){
    double sum = 0.0;
    for(unsigned i = 0; i < prev_layer.size(); i++)
        sum += prev_layer[i].getOutput() * prev_layer[i].weights[neuron_id].weight;
    output = activation(sum);
};

void Neuron::setOutput(double value){ output = value; };

double Neuron::getOutput(){ return output; };

void Neuron::getOutputGradients(double target_value){
    double delta = target_value - output;
    gradient = delta * Neuron::activationPrime(output);
};

void Neuron::updateInputWeights(std::vector<Neuron> &prev_layer){
    for(unsigned i = 0; i <prev_layer.size(); i++){
        double old_delta_weight = prev_layer[i].weights[neuron_id].delta_weight;
        double new_delta_weight = eta * prev_layer[i].getOutput() * gradient + alpha * old_delta_weight;
        prev_layer[i].weights[neuron_id].delta_weight = new_delta_weight;
        prev_layer[i].weights[neuron_id].weight += new_delta_weight;
    }
};

void Neuron::getHiddenGradients(std::vector<Neuron> &next_layer){
    double dow = sumCostDelta(next_layer);
    gradient = dow * Neuron::activation(output);
};

void Neuron::print(){
    std::cout << "\t Neuron " << neuron_id << ": output: " << output << std::endl;
    std::cout << "\t\t weights: ";
    for(unsigned i = 0; i < weights.size(); i++)
        std::cout << weights[i].weight <<" ";
    std::cout << std::endl;
    std::cout << "\t\t weights deltas: ";
    for(unsigned i = 0; i < weights.size(); i++)
        std::cout << weights[i].delta_weight << " ";
    std::cout << std::endl;
};

double Neuron::activation(double x){ return tanh(x); };

double Neuron::activationPrime(double x){ return 1.0 - (x * x); };

double Neuron::sumCostDelta(std::vector<Neuron> &nextLayer) const{
    double sum = 0.0;
    for(unsigned i = 0; i < nextLayer.size(); i++)
        sum += weights[i].weight * nextLayer[i].gradient;
    return sum;
};

double Neuron::generateWeight(void){ return rand() / double(RAND_MAX); };

void Neuron::load(std::vector<double> &weight_data){
    output = weight_data[0];
    for(unsigned i = 0; i < weights.size(); i++){
        weights[i].weight = weight_data[2*i + 1];
        weights[i].delta_weight = weight_data[2*i + 2];
    }
}

void Neuron::save(std::fstream &file){
    file << output;
    for(unsigned i = 0; i < weights.size(); i++)
        file << ", " << weights[i].weight << ", " << weights[i].delta_weight;
}