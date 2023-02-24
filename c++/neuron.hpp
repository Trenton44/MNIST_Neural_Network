class Layer;

#ifndef neuron
#define neuron

struct Connection
{
    double weight;
    double delta_weight;
};

class Neuron {
    public:
        Neuron(unsigned id, unsigned outputs);
        void forwards(std::vector<Neuron> &prevLayer);
        void setOutput(double value);
        double getOutput();
        void getOutputGradients(double target_value);
        void updateInputWeights(std::vector<Neuron> &prevLayer);
        void getHiddenGradients(std::vector<Neuron> &nextLayer);
        void print();

    private:
        unsigned neuron_id;
        double output;
        double gradient;
        static double eta;
        static double alpha;
        std::vector<Connection> weights;
        static double activation(double x);
        static double activationPrime(double x);
        double sumCostDelta(std::vector<Neuron> &nextLayer) const;
        static double generateWeight(void);
};

#endif