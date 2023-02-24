#include <vector>
#include "layer.hpp"
#include "neuron.hpp"

std::vector<Neuron> &Layer::getNeurons() { return neurons; }

Layer::Layer(unsigned id, unsigned size, unsigned connections){
    layer_id = id;
    std::cout << "Creating layer " << layer_id << std::endl;
    for(unsigned i = 0; i < size; i++)
        neurons.push_back(Neuron(i, connections));
    std::cout << "Layer " << layer_id << " completed." << std::endl;
}

void Layer::print(){
    std::cout << "Layer " << layer_id << " (size " << neurons.size() << "):" << std::endl;
    for(unsigned i = 0; i < neurons.size(); i++)
        neurons[i].print();
}

