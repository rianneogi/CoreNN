#pragma once
#include "../Neuron.h"
class MultiplyNeuron : public Neuron
{
public:
	Blob* Input1;
	Blob* Input2;
	Blob* Output;

	MultiplyNeuron();
	MultiplyNeuron(Blob* input1, Blob* input2, Blob* output);
	~MultiplyNeuron();

	void forward();
	void backprop();
	std::vector<Blob*> getVariables();
};
