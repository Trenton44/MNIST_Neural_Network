#include <vector>

class Neuron; // defined in neuron.hpp, this is to let the compiler know it will exist.

#ifndef layer
#define layer

class Layer {
    public:
        Layer(unsigned id, unsigned size, unsigned connections);
        std::vector<Neuron> &getNeurons();
        void print();
    private:
        unsigned layer_id;
        std::vector<Neuron> neurons;
};

#endif