#include <vector>
#include <cstdlib>
#include "layer.hpp"

#ifndef network
#define network
typedef std::vector<double> Sample;
class Network {
    public:
        Network(const std::vector<unsigned> &topology);
        unsigned results(void);
        double train(std::string network_save_filename, const std::vector<Sample> &data, const std::vector<double> target_values, const unsigned epochs, const unsigned sample_count);
        void predict(const std::vector<double> &data);
        bool save(std::string filename);
        static Network load(std::string filename);
        void print();

    private:
        std::vector<Layer> layers;
        double error;
        void forwards(const std::vector<double> &data);
        double backwards(const std::vector<double> &target_values);
};

#endif