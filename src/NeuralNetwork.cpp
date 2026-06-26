//
// Created by firep on 23/06/2026.
//

#include "NeuralNetwork.h"

#include <iostream>
#include <ostream>


bool NeuralNetwork::add_layer(DenseLayer layer) {
	if (is_compiled) {
		std::cerr << "Cannot add new layer after the neural network was compiled" << std::endl;
		return false;
	}
	layers.push_back(std::move(layer));
	return true;
}


bool NeuralNetwork::compile() {
	if (learning_rate > 1 || learning_rate < 0) {
		std::cerr << "ValueError: learning rate must be on the interval [0, 1]" << std::endl;
		return false;
	}

	if (layers.size() < 1) {
		std::cerr << "NeuralNetwork::compile(): layers is empty" << std::endl;
		return false;
	}

	for (int i = 1; i<layers.size(); i++) {
		layers[i].previous = &layers[i-1];
	}
	is_compiled = true;
	return true;
}

static void print_vector(const std::vector<float>& vec) {
	for (auto& v : vec) {
		printf("%f ", v);
	}
	printf("\n");
}

bool NeuralNetwork::forward_propagation(const std::vector<float>& input) {
	if (!is_compiled) {
		std::cerr << "Neural network must be compiled first" << std::endl;
		return false;
	}

	if (layers.empty()) {
		std::cerr << "Network has no layers" << std::endl;
		return false;
	}

	// First DenseLayer consumes the raw input vector
	if (!layers[0].forward(input)) {
		return false;
	}

	// Every next DenseLayer consumes previous layer's output activations
	for (size_t i = 1; i < layers.size(); i++) {
		if (!layers[i].forward(layers[i - 1].activation_values)) {
			return false;
		}
	}

	std::cout << "Output value: "
			  << layers.back().activation_values[0]
			  << std::endl;

	return true;
}


bool NeuralNetwork::fit(const std::vector<float>& X, const std::vector<float>& y) {
	if (!forward_propagation(X)) {
		return false;
	}
	return true;
}

