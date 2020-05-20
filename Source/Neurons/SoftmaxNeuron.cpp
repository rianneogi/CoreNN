#include "Neurons/SoftmaxNeuron.h"

SoftmaxNeuron::SoftmaxNeuron() : Neuron()
{
}

SoftmaxNeuron::SoftmaxNeuron(Blob* input, Blob* output) : mInput(input), mOutput(output)
{
	assert(input->Data.mSize == output->Data.mSize);
}

SoftmaxNeuron::~SoftmaxNeuron()
{
}

bool SoftmaxNeuron::init()
{
#ifdef NN_DEBUG
	assert(mInput->Data.mShape[0] == mOutput->Data.mShape[0]);
	assert(mInput->Data.mSize == mOutput->Data.mSize);
#endif

	uint64_t BatchSize = mInput->Data.mShape[0];
	uint64_t ChannelSize = mInput->Data.mSize / BatchSize;
	cudnnTensorFormat_t TensorFormat = CUDNN_TENSOR_NHWC;

	checkCUDNN(cudnnCreateTensorDescriptor(&mInputDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(mInputDesc,
                                      /*format=*/TensorFormat,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/BatchSize,
                                      /*channels=*/ChannelSize,
                                      /*image_height=*/1,
                                      /*image_width=*/1));

	checkCUDNN(cudnnCreateTensorDescriptor(&mOutputDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(mOutputDesc,
                                      /*format=*/TensorFormat,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/BatchSize,
                                      /*channels=*/ChannelSize,
                                      /*image_height=*/1,
                                      /*image_width=*/1));
}

void SoftmaxNeuron::forward()
{
#ifdef USE_GPU
	forwardGPU();
	// forwardCPU();
#else
	forwardCPU();
#endif
}

void SoftmaxNeuron::backprop()
{
#ifdef USE_GPU
	backpropGPU();
	// backpropCPU();
#else
	backpropCPU();
#endif
}

void SoftmaxNeuron::forwardCPU()
{
	float sum = 0.0f;
	for (uint64_t i = 0; i < mOutput->Data.mSize; i++)
	{
		sum += exp(mInput->Data(i));
	}
	for (uint64_t i = 0; i < mOutput->Data.mSize; i++)
	{
		mOutput->Data(i) = exp(mInput->Data(i))/sum;
	}
}

void SoftmaxNeuron::backpropCPU()
{
	for (uint64_t i = 0; i < mInput->Delta.mSize; i++)
	{
		mInput->Delta(i) += mOutput->Data(i)*(1.0 - mOutput->Data(i));
	}
}

std::vector<Blob*> SoftmaxNeuron::getVariables()
{
	return std::vector<Blob*>();
}