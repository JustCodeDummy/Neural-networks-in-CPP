//
// Created by firep on 23/06/2026.
//

#ifndef NEURALNETWORKSC_NEURALNETWORK_H
#define NEURALNETWORKSC_NEURALNETWORK_H
#include <vector>

#include "DenseLayer.h"


class NeuralNetwork {
	public:
		std::vector<DenseLayer> layers;

		bool compile();
		bool add_layer(DenseLayer layer);
		float learning_rate = .01f;
		bool forward_propagation(const std::vector<float> &input);
	private:
		bool is_compiled = false;
};


#endif //NEURALNETWORKSC_NEURALNETWORK_H
