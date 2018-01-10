#pragma once

#include "../Neuron.h"

class ConvNeuron : public Neuron
{
public:
	Blob* mInput;
	Blob* mOutput;

	uint64_t InputSize;

	uint64_t OutputWidth;
	uint64_t OutputHeight;
	uint64_t OutputDepth;

	uint64_t FieldWidth;
	uint64_t FieldHeight;

	uint64_t BatchSize;

	Blob* Weights;
	Blob* Biases;

	Tensor Ones;

	Float LearningRate;

	ConvNeuron();
	ConvNeuron(Blob* input, Blob* output, Float learning_rate);
	~ConvNeuron();

	void forward();
	void backprop();
	std::vector<Blob*> getVariables();
};

