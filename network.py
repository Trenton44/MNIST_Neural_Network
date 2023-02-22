import csv
import math
from random import random

LEARN_RATE = 0.1
RUNS = 100

formatSample = lambda data: [(float(pixel) / 255.0) for pixel in data]
randomNum = lambda: (random() * 2.00) - 1.00

loss = lambda actual, expected: (expected - actual) ** 2 # Mean Squared Error
loss_prime = lambda actual, expected: 2.00 * (actual - expected) # derivative of Mean Squared Error
activation = lambda x: 1.00 / (1.00 + math.exp(-x)) # sigmoid
activation_prime = lambda x: activation(x) / (1.00 - activation(x)) # deriviative of sigmoid

def softMax(data):
    max = data[0]
    for neuron in data:
        max = neuron if neuron > max else max
    exponented_layer = [ math.exp(neuron - max) for neuron in data ]
    layer_sum = sum(exponented_layer)
    normalized = [ neuron/layer_sum for neuron in exponented_layer ]
    return normalized

class Network:
    def __init__(self, sizes) -> None:
        self.layers = [ 
            Mesh(
                index, 
                sizes[index], 
                sizes[index - 1]
            ) for index in range(1, len(sizes))
        ]

    def forward(self, data):
        self.output = data
        for layer in self.layers:
            self.output = layer.forward(self.output)
        return self.output
    
    def backward(self, expected, output):
        delta_error = [ loss_prime(expected[j], output[j]) for j in range(len(output)) ]
        for index in range(len(self.layers) -1, -1, -1):
            print("Layers: ", self.layers[index].id, self.layers[index - 1].id)
            delta_error = self.layers[index].backward(delta_error, self.layers[index - 1])
        return delta_error

    def train(self, data, answers, epochs):
        sample_size = len(data)
        for i in range(epochs):
            error = 0
            for j in range(sample_size):
                output = data[j]
                output = self.forward(output)
                answer_array = [ 1.00 if a_index == answers[j] else 0.00 for a_index in range(len(output)) ]
                loss_array = [ loss(answer_array[o_index], output[o_index]) for o_index in range(len(output)) ]
                error += sum(loss_array) # answers & data should be matching size
                self.backward(answer_array, output)
            error /= sample_size
            print("Epoch %d/%d error=%f " % (i+1, epochs, error))


class Mesh:
    def __init__(self, id, num_neurons, input_size) -> None:
        self.id = id
        self.connections = [ Neuron(i, input_size) for i in range(num_neurons) ]

    def forward(self, data):
        for connection in self.connections:
            connection.forward(data)
        self.output = [ connection.output[1] for connection in self.connections ]
        return self.output

    def backward(self, delta_error, prev_layer):
        #delta_error is an array of delta_weights from the previous layer
        # iterate over every neuron in the layer
        for index, neuron in enumerate(self.connections):
            delta_activation = activation_prime(neuron.output[1])
            # calculate the change in weight relative to change in cost, for every weight on current neuron
            """
            neuron.delta_weights = [ 
                delta_error[index] * delta_activation * self.connections[w_index].output[1] 
                for w_index in range(len(neuron.weights)) 
            ]
            """
            print(len(prev_layer.connections))
            print(len(neuron.weights))
            print("")
            neuron.delta_weights = [
                delta_error[index] * delta_activation * prev_layer.connections[w_index].output[1]
                for w_index in range(len(neuron.weights))
            ]
            # apply delta_weight to every weight in current neuron
            for w_index in range(len(neuron.weights)):
                neuron.weights[w_index] -= LEARN_RATE * neuron.delta_weights[w_index]
        
        # Calculate the delta_error of each neuron for the previous layer. this will be passed back to that layer.
        weight_gradient = []
        for pn_index, prev_neuron in enumerate(prev_layer.connections):
            total = 0
            for n_index in range(len(self.connections)):
                neuron = self.connections[n_index]
                total += neuron.delta_weights[pn_index] * neuron.weights[pn_index] * (prev_neuron.output[1] * (1 - prev_neuron.output[1]))
                # Note, the activations of prev_neuron might only be multiplied AFTER the sum is calculated, not apart of the sum calc. may be erroneous
            weight_gradient.append(total)
        return weight_gradient

class Neuron:
    def __init__(self, id, num_connections) -> None:
        self.generateWeights = lambda size: [ randomNum() for i in range(size)]
        self.id = id
        self.bias = randomNum()
        self.delta_bias = 0
        self.weights = [ randomNum() for i in range(num_connections) ]
        self.delta_weights = []
        self.output = [ None, None ]
    
    def forward(self, data):
        net_input = 0
        for i in range(len(self.weights)):
            net_input += data[i] * self.weights[i]
        self.output = [ net_input, activation(net_input + self.bias) ]
        return self.output

        
        
    


#forward propagate
# 
#for epochs
#   run all samples in each epoch
#   sample error is loss func
#   delta_error is loss_prime (change in cost with respect to y)
#   start backprop
#       delta error is used with activation_prime
#       return results
#   that delta_error is used to alter weights
# repeat between activation and pre-activation steps
# done

layer_sizes = [784, 16, 16, 10]
NNetwork = Network(layer_sizes)

train_read = csv.reader(open("mnist_train.csv", mode="r"))
train_data = []
train_answer = []

test_data = []
test_read = csv.reader(open("mnist_test.csv", mode="r"))

try:
    while True:
        temp = train_read.__next__()
        train_answer.append(float(temp[0]))
        train_data.append(formatSample(temp[1:]))
        test_data.append(formatSample(test_read.__next__()))
except StopIteration:
    pass

NNetwork.train(train_data[0: 1000], train_answer[0:1000], 35)


"""
take delta cost
put though activation prime to get change of output with respect to input

calculate each weight relative to delta cost
"""