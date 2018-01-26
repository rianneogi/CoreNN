#pragma once
#include "Neuron.h"
class ReshapeNeuron :
	public Neuron
{
public:
	Blob* mInput;

	TensorShape InputShape;
	TensorShape OutputShape;

	ReshapeNeuron();
	ReshapeNeuron(Blob* input, TensorShape output_shape);
	~ReshapeNeuron();

	void forward();
	void backprop();
	std::vector<Blob*> getVariables();
	void reset();
};
