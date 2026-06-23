cmake_minimum_required(VERSION 4.0)
project(NeuralNetworksC)

set(CMAKE_CXX_STANDARD 20)

add_executable(NeuralNetworksC main.cpp src/Matrix.cpp src/Matrix.h
        src/DenseLayer.cpp
        src/DenseLayer.h
        src/Neuron.cpp
        src/Neuron.h
        src/NeuralNetwork.cpp
        src/NeuralNetwork.h)
