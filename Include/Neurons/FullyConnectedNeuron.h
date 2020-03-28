#pragma once

#include "../Neuron.h"

class FullyConnectedNeuron : public Neuron
{
public:
	Blob* mInput;
	Blob* mOutput;

	Initializer* mInitializer;

	uint64_t InputSize;
	uint64_t OutputSize;
	uint64_t BatchSize;
	Blob* Weights;
	Blob* Biases;
	// Blob* BiasesStacked;
	/*Tensor WeightsDelta;
	Tensor BiasesDelta;*/
	Tensor Ones;

	FullyConnectedNeuron();
	// FullyConnectedNeuron(Blob* input, Blob* output);
	FullyConnectedNeuron(Blob* input, Blob* output, Initializer* initializer);
	~FullyConnectedNeuron();

	bool init();

	void forward();
	void backprop();
	std::vector<Blob*> getVariables();
};
