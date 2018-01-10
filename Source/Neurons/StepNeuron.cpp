#include "Neurons/StepNeuron.h"

StepNeuron::StepNeuron() : Neuron()
{
}

StepNeuron::StepNeuron(Blob* input, Blob* output) : mInput(input), mOutput(output)
{
	assert(input->Data.mSize == output->Data.mSize);
}

StepNeuron::~StepNeuron()
{
}

void StepNeuron::forward()
{
	for (uint64_t i = 0; i < mOutput->Data.mSize; i++)
	{
		mOutput->Data(i) = mInput->Data(i) > 0 ? 1.0 : 0.0;
	}
}

void StepNeuron::backprop()
{
	for (uint64_t i = 0; i < mInput->Delta.mSize; i++)
	{
		mInput->Delta(i) += 1.0;
	}
}

std::vector<Blob*> StepNeuron::getVariables()
{
	return std::vector<Blob*>();
}
