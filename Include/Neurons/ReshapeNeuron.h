#pragma once
#include "..\Neuron.h"
class ReshapeNeuron :
	public Neuron
{
public:
	Blob* mInput;
	Blob* mOutput;

	TensorShape InputShape;
	TensorShape OutputShape;

	ReshapeNeuron();
	ReshapeNeuron(Blob* input, Blob* output, TensorShape output_shape);
	~ReshapeNeuron();

	void forward();
	void backprop();
	std::vector<Blob*> getVariables();
};

