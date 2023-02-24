#include <vector>
#include <cstdlib>
#include "layer.hpp"

#ifndef network
#define network
typedef std::vector<double> Sample;
class Network {
    public:
        Network(const std::vector<unsigned> &topology);
        void results(std::vector<double> &result_values);
        void train(const std::vector<Sample> &data, const std::vector<double> target_values, const unsigned epochs);
        void print();
    private:
        std::vector<Layer> layers;
        double error;
        void forwards(const std::vector<double> &data);
        void backwards(const std::vector<double> &target_values);
};


#endif