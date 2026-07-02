//
// Created by firep on 16/06/2026.
//

#include "DenseLayer.h"

#include <cmath>
#include <iostream>
#include <random>


DenseLayer::DenseLayer(size_t n_in, size_t n_out): in(n_in), out(n_out) {
	weights.resize(n_out * n_in);
	bias.resize(n_out);
	grad_weights.resize(n_out * n_in);
	grad_bias.resize(n_out);
	grad_neurons.resize(n_out);
	activation_values.resize(n_out);


	// TODO move to appropriate function/place
	/*switch (activationFunction) {
		case ACTIVATION_FUNCTION::SIGMOID:
		case ACTIVATION_FUNCTION::TANH:
			sig_weight_initialization();
			break;
		case ACTIVATION_FUNCTION::RELU:
		default:
			relu_weight_initialization();
			break;
	}*/

}






float DenseLayer::activation(float value) {
	// TODO add other activation functions as well
	switch (activationFunction) {
		case ACTIVATION_FUNCTION::SIGMOID:
			return sig_(value);
		case ACTIVATION_FUNCTION::TANH:
			return tanh_(value);
		case ACTIVATION_FUNCTION::RELU:
		default:
			return relu_(value);
	}
}

static void print_vector(const std::vector<float>& vec) {
	for (auto& v : vec) {
		printf("%f ", v);
	}
	printf("\n");
}









bool DenseLayer::forward(const std::vector<float>& previous_activations) {
	if (previous_activations.size() != in) {
		std::cerr << "Input size mismatch. Expected "
				  << in << ", got " << previous_activations.size() << std::endl;
		return false;
	}

	if (weights.size() != in * out) {
		std::cerr << "Weight size mismatch. Expected "
				  << in * out << ", got " << weights.size() << std::endl;
		return false;
	}

	if (bias.size() != out) {
		std::cerr << "Bias size mismatch\n";
		return false;
	}

	activation_values.resize(out);

	for (size_t neuron = 0; neuron < out; neuron++) {
		float val = bias[neuron];

		for (size_t input = 0; input < in; input++) {
			size_t w = neuron * in + input;
			val += previous_activations[input] * weights[w];
		}

		activation_values[neuron] = activation(val);
	}

	return true;
}

void DenseLayer::relu_weight_initialization() {
	const float stddev = std::sqrt(2.0f / static_cast<float>(in));

	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<float> dist(0.0f, stddev);

	for (float& w : weights) {
		w = dist(gen);
	}
	for (float& b : bias) {
		b = 0.0f;
	}
}

void DenseLayer::sig_weight_initialization() {
	const float limit = std::sqrt(6.0f / static_cast<float>(in + out));

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist(-limit, limit);

	for (float& w : weights) {
		w = dist(gen);
	}

	for (float& b : bias) {
		b = 0.0f;
	}
}

