#pragma once

#include "../Neuron.h"

class ConvNeuron : public Neuron
{
public:
	Blob* mInput;
	Blob* mOutput;

	uint64_t InputWidth;
	uint64_t InputHeight;
	uint64_t InputDepth;

	uint64_t OutputWidth;
	uint64_t OutputHeight;
	uint64_t OutputDepth;

	uint64_t FieldWidth;
	uint64_t FieldHeight;
	uint64_t PaddingX;
	uint64_t PaddingY;
	uint64_t StrideX;
	uint64_t StrideY;
	uint64_t DilationX;
	uint64_t DilationY;

	uint64_t BatchSize;

	Blob* Weights;
	Blob* Biases;

	Tensor Ones;

	Float LearningRate;

	cudnnTensorDescriptor_t InputDesc;
	cudnnTensorDescriptor_t OutputDesc;
	cudnnFilterDescriptor_t FilterDesc;
	cudnnConvolutionDescriptor_t ConvDesc;
	cudnnConvolutionFwdAlgo_t ForwardAlg;
	size_t ForwardWorkspaceBytes;
	void *dForwardWorkspace;

	ConvNeuron();
	ConvNeuron(Blob* input, Blob* output, int filter_x, int filter_y, int pad_x, int pad_y, int stride_x, int stride_y, int dilation_x=1, int dilation_y=1);
	~ConvNeuron();

	void forward();
	void forwardGPU();
	void backprop();
	void backpropGPU();
	std::vector<Blob *> getVariables();
};

