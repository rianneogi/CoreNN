#pragma once
#include "../Neuron.h"
class AddNeuron : public Neuron
{
public:
	Blob* Input1;
	Blob* Input2;
	Blob* Output;

	AddNeuron();
	AddNeuron(Blob* input1, Blob* input2, Blob* output);
	~AddNeuron();

	void forward();
	void backprop();
	std::vector<Blob*> getVariables();
};
