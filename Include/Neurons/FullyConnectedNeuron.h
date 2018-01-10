#pragma once

#include "../Neuron.h"

class FullyConnectedNeuron : public Neuron
{
public:
	Blob* mInput;
	Blob* mOutput;

	uint64_t InputSize;
	uint64_t OutputSize;
	uint64_t BatchSize;
	Blob* Weights;
	Blob* Biases;
	/*Tensor WeightsDelta;
	Tensor BiasesDelta;*/
	Tensor Ones;

	FullyConnectedNeuron(); 
	FullyConnectedNeuron(Blob* input, Blob* output);
	FullyConnectedNeuron(Blob* input, Blob* output, Float init_start, Float init_end);
	~FullyConnectedNeuron();

	void forward();
	void backprop();
	std::vector<Blob*> getVariables();
};

