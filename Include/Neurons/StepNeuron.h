#pragma once

#include "../Neuron.h"

class StepNeuron : public Neuron
{
public:
	Blob* mInput;
	Blob* mOutput;

	StepNeuron();
	StepNeuron(Blob* input, Blob* output);
	~StepNeuron();

	void forward();
	void backprop();
	std::vector<Blob*> getVariables();
};
