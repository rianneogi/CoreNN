#include "Neurons/TanhNeuron.h"

TanhNeuron::TanhNeuron() : Neuron()
{
}

TanhNeuron::TanhNeuron(Blob* input, Blob* output) : mInput(input), mOutput(output)
{
}

TanhNeuron::~TanhNeuron()
{
}

bool TanhNeuron::init()
{
	assert(mInput->Data.mSize == mOutput->Data.mSize);

	return true;
}

void TanhNeuron::forward()
{
	for (uint64_t i = 0; i < mOutput->Data.mSize; i++)
	{
		mOutput->Data(i) = tanh(mInput->Data(i));
	}
}

void TanhNeuron::backprop()
{
	for (uint64_t i = 0; i < mInput->Delta.mSize; i++)
	{
		mInput->Delta(i) += mOutput->Delta(i)*(1.0 - mOutput->Data(i)*mOutput->Data(i));
	}
}

std::vector<Blob*> TanhNeuron::getVariables()
{
	return std::vector<Blob*>();
}
