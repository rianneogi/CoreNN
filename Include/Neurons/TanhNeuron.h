#pragma once

#include "../Neuron.h"

//Fully connected Tanh Neuron
//	Input: Matrix: Batch of input vectors
//  Output: Vector: Tanh nonlinearity for each of the inputs

//Forward:
//	O = tanh(W*I)

//Backward:
//  BO = (W^t*D)x(1-O^2)

class TanhNeuron : public Neuron
{
public:
	Blob* mInput;
	Blob* mOutput;

	TanhNeuron();
	TanhNeuron(Blob* input, Blob* output);
	~TanhNeuron();

	bool init();

	void forward();
	void backprop();
	std::vector<Blob*> getVariables();
};
