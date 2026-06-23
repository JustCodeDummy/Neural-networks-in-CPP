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

bool NeuralNetwork::forward_propagation(const std::vector<float> &input) {
	if (!is_compiled) {
		std::cerr << "Neural network must be compiled first";
		return false;
	}

	layers[0].activation_values = input;
	for (auto& layer : layers) {
		layer.forward();
	}

	return true;
}