#pragma once

#include "../Neuron.h"

class DiagNeuron : public Neuron
{
public:
	Blob* mInput;
	Blob* mOutput;

	uint64_t InputSize;

	uint64_t InputHeight;
	uint64_t InputWidth;
	uint64_t InputDepth;

	uint64_t OutputCols;
	uint64_t OutputRows;

	uint64_t BatchSize;
	uint64_t DiagCount;

	Tensor PadValue;

	DiagNeuron();
	DiagNeuron(Blob* input, Blob* output, Tensor pad_value);
	~DiagNeuron();

	void forward();
	void backprop();
	std::vector<Blob*> getVariables();
};