from NeuralNetwork import neuralNetwork

input_nodes = 3
hidden_nodes = 3
output_nodes = 3

learning_rate = 0.3


def run():
    nn = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    print("weight matrice input layer:\n", nn.wih, "\n\n", "weight matrice output layer:\n", nn.who)

    # matplotlib.pyplot.imshow(nn.wih, interpolation="nearest")

    output_signals = nn.query([1.0, 0.5, -1.5])

    print("\noutput_signals:\n", output_signals)

    pass


run()
