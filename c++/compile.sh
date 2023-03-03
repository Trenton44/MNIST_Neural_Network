echo "Compiling C++ program..."
g++ main.cpp -o network -Wall -Wextra -Werror && echo "succesfully compiled."
echo ""
./network

# emcc <web.cpp> -o network.js -s NO_EXIT_RUNTIME=1 -s EXPORTED_RUNTIME_METHODS=[ccall] -s NO_DISABLE_EXCEPTION_CATCHING -s ALLOW_MEMORY_GROWTH -s MAXIMUM_MEMORY=4GB --embed-file /home/tmc069/MNIST_Neural_Network/c++/network_save.csv