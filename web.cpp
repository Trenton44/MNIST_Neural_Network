#include <vector>
#include <iostream>
#include <fstream>
#include <emscripten/emscripten.h>

#include "csv.cpp"
#include "neuron.cpp"
#include "layer.cpp"
#include "network.cpp"

typdef std::vector<double> Sample;
Network net = Newtork::load("");

int main(){
    std::cout << "Neural Network successfully loaded." << std::endl;
}

#ifdef _cplusplus
#define EXTERN extern "C"
#else
#define EXTERN
#endif

EXTERN EMSCRIPTEN_KEEPALIVE unsigned readNumber(){
    //net.predict();
    return net.results();
}