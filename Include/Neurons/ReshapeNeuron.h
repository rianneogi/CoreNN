#pragma once
#include "Neuron.h"
class ReshapeNeuron :
	public Neuron
{
public:
	Blob* mInput;

	TensorShape InputShape;
	TensorShape InputSubshape;
	TensorShape InputOffset;
	
	TensorShape OutputShape;
	TensorShape OutputSubshape;
	TensorShape OutputOffset;

	ReshapeNeuron();
	ReshapeNeuron(Blob* input, TensorShape output_shape);
	ReshapeNeuron(Blob* input, TensorShape output_shape, TensorShape output_offset, TensorShape output_subshape);
	~ReshapeNeuron();

	bool init();
	void forward();
	void backprop();
	std::vector<Blob*> getVariables();
	void reset();
};
