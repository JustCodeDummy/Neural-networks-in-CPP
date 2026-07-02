//
// Created by firep on 23/06/2026.
//

#ifndef NEURALNETWORKSC_NEURALNETWORK_H
#define NEURALNETWORKSC_NEURALNETWORK_H
#include <vector>

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
		bool compile();
		bool add_layer(DenseLayer layer);
		float learning_rate = .01f;
		bool forward_propagation(const std::vector<float> &input, std::vector<float>& output);
		bool fit(const std::vector<float>& X, const std::vector<float>& y);
		bool back_propagation(std::vector<float>& y);


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

};


#endif //NEURALNETWORKSC_NEURALNETWORK_H
