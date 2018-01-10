#pragma once
#include "../Neuron.h"

class KnightNeuron : public Neuron
{
public:
	Blob* mInput;
	Blob* mOutput;

	uint64_t InputWidth;
	uint64_t InputHeight;
	uint64_t InputDepth;

	uint64_t OutputCols;
	uint64_t OutputRows;

	uint64_t FieldCount;

	uint64_t BatchSize;

	Tensor PadValue;

	KnightNeuron();
	KnightNeuron(Blob* input, Blob* output, int64_t field_width, int64_t field_height, Tensor pad_value);
	~KnightNeuron();

	void forward();
	void backprop();
	std::vector<Blob*> getVariables();
};
