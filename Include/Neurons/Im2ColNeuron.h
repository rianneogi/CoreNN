#pragma once

#include "Neuron.h"

//Converts a 4D Tensor into a 2D Tensor for matrix multiplication
//Input: BatchSize x InputHeight x InputWidth x InputDepth
//Output: BatchSize*(InputWidth - FieldWidth + 1)*(InputHeight - FieldHeight + 1) x FieldWidth*FieldHeight*InputDepth

//WARNING: Works only for odd sized receptive fields

class Im2ColNeuron : public Neuron
{
public:
	Blob* mInput;
	Blob* mOutput;

	uint64_t InputWidth;
	uint64_t InputHeight;
	uint64_t InputDepth;

	uint64_t OutputCols;
	uint64_t OutputRows;

	uint64_t FieldWidth;
	uint64_t FieldHeight;

	uint64_t FieldCount;

	uint64_t BatchSize;

	Im2ColNeuron();
	Im2ColNeuron(Blob* input, Blob* output, uint64_t field_width, uint64_t field_height);
	~Im2ColNeuron();

	bool init();

	void forward();
	void backprop();
	std::vector<Blob*> getVariables();
};
