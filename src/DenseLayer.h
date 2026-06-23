//
// Created by firep on 16/06/2026.
//

#ifndef NEURALNETWORKSC_DENSELAYER_H
#define NEURALNETWORKSC_DENSELAYER_H
#include <cstddef>
#include <vector>

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

		std::vector<float> grad_weights; // out x in
		std::vector<float> grad_bias;    // out

		std::vector<float> activation_values;
		bool forward() const;
		std::vector<float> delta;

	private:
		void relu_weight_initialization();
		void sig_weight_initialization();


};









#endif //NEURALNETWORKSC_DENSELAYER_H