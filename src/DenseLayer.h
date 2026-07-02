#pragma once

//
// Created by firep on 16/06/2026.
//

#define NEURALNETWORKSC_DENSELAYER_H
#include <cstddef>
#include <vector>
#include <cmath>
#include "STATUS.h"
enum class ACTIVATION_FUNCTION {
	RELU = 0,
	SIGMOID = 1,
	TANH = 2,
};



class DenseLayer {
	public:
		std::size_t in;
		std::size_t out;
		DenseLayer(size_t n_in, size_t n_out);
		ACTIVATION_FUNCTION activationFunction = ACTIVATION_FUNCTION::RELU;
		DenseLayer* previous = nullptr;
		std::vector<float> weights;      // out x in
		std::vector<float> bias;         // out
		std::vector<float> grad_neurons; // out
		std::vector<float> grad_weights; // out x in
		std::vector<float> grad_bias;    // out
		std::vector<float> activation_values;
		bool forward(const std::vector<float>& previous_activations);
		std::vector<float> delta;


	private:
		void relu_weight_initialization();
		void sig_weight_initialization();
		float activation(float value);
		static float sig_(float value) { return 1 / (1 + exp(-value)); }
		static float tanh_(float value) { return (exp(value) - exp(-value)) / (exp(value) + exp(-value)); }
		static float relu_(float value) { return value < 0 ? 0 : value; }

	/*

	 */

};
