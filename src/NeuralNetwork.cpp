//
// Created by firep on 23/06/2026.
//

#include "NeuralNetwork.h"

#include <iostream>
#include <ostream>


STATUS NeuralNetwork::add_layer(DenseLayer layer) {
	if (is_compiled) {
		std::cerr << "Cannot add new layer after the neural network was compiled" << std::endl;
		return STATUS::ALREADY_COMPILED;
	}
	layers.push_back(std::move(layer));
	return STATUS::OK;
}


STATUS NeuralNetwork::compile() {
	if (learning_rate > 1 || learning_rate < 0) {
		std::cerr << "ValueError: learning rate must be on the interval [0, 1]" << std::endl;
		return STATUS::VALUE_OUT_OF_RANGE_ERROR;
	}

	if (layers.size() < 1) {
		std::cerr << "NeuralNetwork::compile(): layers is empty" << std::endl;
		return STATUS::MISCONFIGURATION;
	}

	for (int i = 1; i<layers.size(); i++) {
		layers[i].previous = &layers[i-1];
	}
	is_compiled = true;
	return STATUS::OK;
}

static void print_vector(const std::vector<float>& vec) {
	for (auto& v : vec) {
		printf("%f ", v);
	}
	printf("\n");
}

STATUS NeuralNetwork::forward_propagation(const std::vector<float>& input, std::vector<float>& outarr) {
	if (!is_compiled) {
		std::cerr << "Neural network must be compiled first" << std::endl;
		return STATUS::NOT_COMPILED;
	}

	if (layers.empty()) {
		std::cerr << "Network has no layers" << std::endl;
		return STATUS::MISCONFIGURATION;
	}

	// First DenseLayer consumes the raw input vector
	if (!layers[0].forward(input)) {
		return STATUS::PROPAGATION_FAILED;
	}

	// Every next DenseLayer consumes previous layer's output activations
	for (size_t i = 1; i < layers.size(); i++) {
		if (!layers[i].forward(layers[i - 1].activation_values)) {
			return STATUS::BACKPROPAGATE_FAILED; // TODO add correct STATUS value
		}
	}

	std::cout << "Output value: "
			  << layers.back().activation_values[0]
			  << std::endl;

	outarr = layers.back().activation_values;

	return STATUS::OK;
}

float NeuralNetwork::Loss_(float x, float y) {
	switch (lossFunction) {
		case LOSS_FUNCTION::MSE:
			return mse_(x, y);
		case LOSS_FUNCTION::RMSE:
		default:
			return x-y;
	}
}

float NeuralNetwork::derivative(float value, const ACTIVATION_FUNCTION& activationFunction) {
	switch (activationFunction) {
		case ACTIVATION_FUNCTION::SIGMOID:
			return sig_derivative(value);
		case ACTIVATION_FUNCTION::RELU:
			return relu_derivative(value);
		case ACTIVATION_FUNCTION::TANH:
			return tanh_derivative(value);
		default:
			std::cerr << "Unknown activation function"  << std::endl;
			return 0;
	}
}

// TODO multiple loss and activation functions, currently only MSE and sigmoid
STATUS NeuralNetwork::back_propagation(std::vector<float>& y) {

	if (!is_compiled) {
		std::cerr << "Neural network must be compiled first" << std::endl;
		return STATUS::NOT_COMPILED;
	}

	if (layers.back().activation_values.size() != y.size()) {
		std::cerr << "Output layer and expected values array size do not match!" << std::endl;
		return STATUS::SIZE_MISMATCH;
	}

	size_t lidx  = layers.size()-1;
	while (lidx > 0) {

		// if output layer
		if (lidx == layers.size()-1) {
			for (size_t neuron = 0; neuron < layers[lidx].activation_values.size(); neuron++) {
				// assuming sig
				float out = layers[lidx].activation_values[neuron];
				float err = mse_derivative(out, y[neuron]);
				float der = derivative(out, layers[lidx].activationFunction);
				// derivative of loss function * output value * derivative of activation function * learning_rate
				float grad = out * err * der; // * learning_rate;
				layers[lidx].grad_neurons[neuron] = grad;
			}
		}else { // hidden layers
			int nc = layers[lidx+1].activation_values.size(); // next layer neuron count
			int cc = layers[lidx].activation_values.size(); // current layer neuron count
			for (size_t neuron=0; neuron < cc; neuron++) {
				float grad = .0f;
				float a_der =  derivative(layers[lidx].activation_values[neuron], layers[lidx].activationFunction);

				for (size_t next = 0; next < nc; next++) {
					size_t widx = neuron + cc * next;
					grad += layers[lidx+1].grad_neurons[next] * layers[lidx].weights[widx];
				}
				layers[lidx].grad_neurons[neuron] = grad * a_der;
			}
		}
		lidx--;
	}
	return STATUS::OK;

}


STATUS NeuralNetwork::fit(const std::vector<float>& X, const std::vector<float>& y) {
	std::vector<float> output;
	if (forward_propagation(X, output) != STATUS::OK) {
		return STATUS::PROPAGATION_FAILED;
	}

	if (y.size() != output.size()) {
		std::cerr << "Size mismatch output and expected vectors do not match" << std::endl;
		return STATUS::SIZE_MISMATCH;
	}

	if (back_propagation(output) != STATUS::OK) {
		std::cerr << "Backpropagation failed" << std::endl;
		return STATUS::BACKPROPAGATE_FAILED;
	};

	return STATUS::OK;
}

