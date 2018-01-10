#pragma once

#include "Neuron.h"

//Converts a 2D Tensor into a 4D Tensor for matrix multiplication, reverse of Im2Col
//Input: BatchSize*(InputWidth - FieldWidth + 1)*(InputHeight - FieldHeight + 1) x FieldWidth*FieldHeight*InputDepth
//Output: BatchSize x InputHeight x InputWidth x InputDepth

//WARNING: Works only for odd sized receptive fields

class Col2ImNeuron : public Neuron
{
public:
	Blob* mInput;
	Blob* mOutput;

	uint64_t InputCols;
	uint64_t InputRows;

	uint64_t OutputWidth;
	uint64_t OutputHeight;
	uint64_t OutputDepth;

	uint64_t FieldWidth;
	uint64_t FieldHeight;

	uint64_t FieldCount;

	uint64_t BatchSize;

	Col2ImNeuron();
	Col2ImNeuron(Blob* input, Blob* output, uint64_t field_width, uint64_t field_height);
	~Col2ImNeuron();

	void forward();
	void backprop();
	std::vector<Blob*> getVariables();
};
