//
// Created by firep on 16/06/2026.
//

#include "DenseLayer.h"

#include <cmath>
#include <random>

DenseLayer::DenseLayer(size_t n_in, size_t n_out): in(n_in), out(n_out) {
	weights.resize(n_out * n_in);
	bias.resize(n_out);
	grad_weights.resize(n_out * n_in);
	grad_bias.resize(n_out);

	activation_values.resize(n_out);
	for (auto& b : activation_values) {
		b = 0;
	}

	// TODO move to appropriate function/place
	switch (activationFunction) {
		case ACTIVATION_FUNCTION::SIGMOID:
		case ACTIVATION_FUNCTION::TANH:
			sig_weight_initialization();
			break;
		case ACTIVATION_FUNCTION::RELU:
		default:
			relu_weight_initialization();
			break;
	}

}

float activation(float value) {
	// TODO add other activation functions as well
	return value < 0 ? 0 : value; // ReLU
}


//
bool DenseLayer::forward() {

	if (!previous) return true; // input layer

	for (size_t neuron = 0; neuron < out; neuron++) {
		float val = bias[neuron];

		for (size_t input = 0; input < in; input++) {
			size_t w = input + neuron * in;
			val += previous->activation_values[input] * weights[w];
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

