#pragma once
#include "../Neuron.h"

class FileNeuron : public Neuron
{
public:
	Blob* mInput;
	Blob* mOutput;

	uint64_t InputWidth;
	uint64_t InputHeight;
	uint64_t InputDepth;

	uint64_t OutputCols;
	uint64_t OutputRows;

	uint64_t BatchSize;

	FileNeuron();
	FileNeuron(Blob* input, Blob* output);
	~FileNeuron();

	void forward();
	void backprop();
	std::vector<Blob*> getVariables();
};
