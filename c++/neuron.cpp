#include <vector>
#include <iostream>
#include <cmath>
#include "layer.hpp"
#include "neuron.hpp"

double Neuron::learn_rate = 0.15;
double Neuron::alpha = 0.5;

double Neuron::generateWeight(void){ return rand() / double(RAND_MAX); };

double Neuron::activation(double x){ return tanh(x); };
double Neuron::activationPrime(double x){ return 1.0 - (x * x); };

void Neuron::setOutput(double value){ output = value; };
double Neuron::getOutput(){ return output; };

// NOTE: the dC/dW of weights in the network is dC/dW = (dC/dA) * (dA/dZ) * (dZ/dW)

// Get the (dC/dA) * (dA/dZ) of the delta cost function for this neuron
void Neuron::getOutputGradients(double target_value){
    double delta = 2 * (target_value - output); // dC/dA for output layer neurons (derivative of cost function)
    gradient = delta * Neuron::activationPrime(output); // dC/dZ = (dC/dA) * (dC/dZ)
};
// see getOutputGradients
void Neuron::getHiddenGradients(std::vector<Neuron> &next_layer){
    double dow = sumCostDelta(next_layer); // dC/dA: hidden layers use sum of (dC/dZ) values from every neuron in front of it.
    gradient = dow * Neuron::activationPrime(output); // dC/dZ = (dC/dA) * (dC/dZ)
};

// Hidden neurons get their dC/dA by adding the dC/dZ of every neuron in front of it. this function calculates that for this neuron
double Neuron::sumCostDelta(std::vector<Neuron> &next_layer) const {
    double sum = 0.0;
    for(unsigned i = 0; i < next_layer.size(); i++)
        sum += weights[i].weight * next_layer[i].gradient; //dC/dA += weight of neuron connected in front * dC/dZ of front neuron
    return sum;
};

Neuron::Neuron(unsigned id, unsigned outputs){
    output = 0;
    gradient = 0;
    neuron_id = id;
    for(unsigned i = 0; i < outputs; i++){
        weights.push_back(Connection());
        weights.back().weight = generateWeight();
    }
    bias = generateWeight();
}

void Neuron::forwards(std::vector<Neuron> &prev_layer){
    double sum = 0.0;
    for(unsigned i = 0; i < prev_layer.size(); i++)
        sum += prev_layer[i].getOutput() * prev_layer[i].weights[neuron_id].weight;
    output = activation(sum + bias);
};

// getOutputGradients, getInputGradients, & sumCostDelta completed the first two parts of the dC/dW equation.
// now we complete that equation for every weight in this neuron and update them.
// NOTE: dC/dW = (dC/dA) * (dA/dZ) * (dZ/dW)
void Neuron::updateInputWeights(std::vector<Neuron> &prev_layer){
    for(unsigned i = 0; i <prev_layer.size(); i++){
        // the last factor in the dC/dW equation, (dZ/dW) is simply the output of the previous neuron
        double deltaW = gradient * prev_layer[i].getOutput(); // gradient = (dC/dA) * (dA/dZ), dz/dW = output
        // we get the new weight by doing: current weight - (learn_rate * dC/dW)
        double new_weight = learn_rate * deltaW;
        /*
            Now you could stop there, but if you wanted to make the gradient descent a little more careful,
            you can add the old delta weight into the equation, times a factor (alpha) to curve how quickly the weight changes in either direction
            I'll include the code here, but commented out:
            double old_delta_weight = prev_layer[i].weights[neuron_id].delta_weight;
            double curve = alpha * old_delta_weight;
            new_weight += curve;
        */
        prev_layer[i].weights[neuron_id].delta_weight = new_weight;
        prev_layer[i].weights[neuron_id].weight += prev_layer[i].weights[neuron_id].delta_weight;
    }

    // we'll also update the bias in this function.
    // the dC/dB is exactly like the dC/dW function, except we swap (dZ/dW) -> (dZ/dB).
    // and fortunately, dZ/dB is always 1, since the bias is a constant number.
    // which means dC/dB = (dC/dA) * (dA/dZ), which is just dC/dB = gradient
    bias += learn_rate * gradient;
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

void Neuron::save(std::fstream &file){
    file << output;
    for(unsigned i = 0; i < weights.size(); i++)
        file << ", " << weights[i].weight << ", " << weights[i].delta_weight;
};

void Neuron::load(std::vector<double> &weight_data){
    output = weight_data[0];
    for(unsigned i = 0; i < weights.size(); i++){
        weights[i].weight = weight_data[2*i + 1];
        weights[i].delta_weight = weight_data[2*i + 2];
    }
};