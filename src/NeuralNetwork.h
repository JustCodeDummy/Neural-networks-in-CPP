#pragma once
//
// Created by firep on 23/06/2026.
//

#include <vector>
#include "STATUS.h"
#include "DenseLayer.h"

enum class LOSS_FUNCTION {
	MSE,
	RMSE
};


class NeuralNetwork {
	public:
		std::vector<DenseLayer> layers;
		LOSS_FUNCTION lossFunction = LOSS_FUNCTION::MSE;
		NeuralNetwork(){};
		STATUS compile();
		STATUS add_layer(DenseLayer layer);
		float learning_rate = .01f;
		STATUS forward_propagation(const std::vector<float> &input, std::vector<float>& output);
		STATUS fit(const std::vector<float>& X, const std::vector<float>& y);
		STATUS back_propagation(const std::vector<float>& X, const std::vector<float>& y);


	private:
		bool is_compiled = false;
		float Loss_(float x, float y);
		static float mse_(float x, float y)  {return std::pow(y - x, 2)/2; };
		static float mse_derivative(float x, float y) {return x - y;}
		static float relu_derivative(float value) {return value > 0 ? 1 : 0;};
		static float sig_(float value) { return 1 / (1 + exp(-value)); }
		static float tanh_derivative(float value) {return 1  - std::pow(std::tanh(value), 2);};
		static float sig_derivative(float value) {float s = sig_(value); return s * (1 - s); };
		float derivative(float value, const ACTIVATION_FUNCTION& activationFunction);
		bool update_weights(DenseLayer& current, const std::vector<float>& X);
};


