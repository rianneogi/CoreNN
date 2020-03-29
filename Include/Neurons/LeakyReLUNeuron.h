#pragma once
#include "Neuron.h"
class LeakyReLUNeuron :
	public Neuron
{
public:
	Blob* mInput;
	Blob* mOutput;

	Float LeakFactor;

	LeakyReLUNeuron();
	LeakyReLUNeuron(Blob* input, Blob* output, Float leak_factor = 0);
	~LeakyReLUNeuron();

	bool init();

	void forward();
	void forwardGPU();
	void forwardCPU();
	void backprop();
	void backpropGPU();
	void backpropCPU();
	std::vector<Blob *> getVariables();
};

__global__ void leaky_relu_forward(int size, float leak_factor, float *input_data, float* output_data);
__global__ void leaky_relu_backprop(int size, float leak_factor, float *input_data, float *input_delta, float* output_data, float* output_delta);