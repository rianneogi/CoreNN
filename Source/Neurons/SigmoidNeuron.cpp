#include "Neurons/SigmoidNeuron.h"

SigmoidNeuron::SigmoidNeuron() : Neuron()
{
}

SigmoidNeuron::SigmoidNeuron(Blob* input, Blob* output) : mInput(input), mOutput(output)
{
	assert(input->Data.mSize == output->Data.mSize);
}

SigmoidNeuron::~SigmoidNeuron()
{
}

void SigmoidNeuron::forward()
{
	for (uint64_t i = 0; i < mOutput->Data.mSize; i++)
	{
		mOutput->Data(i) = sigmoid(mInput->Data(i));
	}
}

void SigmoidNeuron::backprop()
{
	for (uint64_t i = 0; i < mInput->Delta.mSize; i++)
	{
		mInput->Delta(i) += mOutput->Delta(i)*mOutput->Data(i)*(1.0 - mOutput->Data(i));
	}
}

std::vector<Blob*> SigmoidNeuron::getVariables()
{
	return std::vector<Blob*>();
}
