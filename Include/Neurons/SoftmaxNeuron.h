#pragma once

#include "../Neuron.h"

//Softmax Neuron
//	Input: Matrix: Batch of input vectors
//  Output: Vector: Softmax on each of the inputs

//Converts values into probabilities

//Forward:
//	O[i] = exp(i)/(sum_{j=1}^n exp(j)) , take exponentials of the values and normalize so that sum_{i=1]^n O[i] = 1

//Backward:
//  BO[i] = O[i](1-O[i])

class SoftmaxNeuron : public Neuron
{
public:
	Blob* mInput;
	Blob* mOutput;

	// cudnnSoftmaxAlgorithm_t mAlgorithm;
	cudnnTensorDescriptor_t mInputDesc;
	cudnnTensorDescriptor_t mOutputDesc;

	SoftmaxNeuron();
	SoftmaxNeuron(Blob* input, Blob* output);
	~SoftmaxNeuron();

	bool init();
	void forward();
	void forwardCPU();
	void forwardGPU();
	void backprop();
	void backpropCPU();
	void backpropGPU();
	std::vector<Blob*> getVariables();
};