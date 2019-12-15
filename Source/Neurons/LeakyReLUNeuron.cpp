#include "Neurons/LeakyReLUNeuron.h"

#include <algorithm>

LeakyReLUNeuron::LeakyReLUNeuron()
{
}

LeakyReLUNeuron::LeakyReLUNeuron(Blob* input, Blob* output, Float leak_factor)
	: mInput(input), mOutput(output), LeakFactor(leak_factor)
{
}

LeakyReLUNeuron::~LeakyReLUNeuron()
{
}

bool LeakyReLUNeuron::init()
{
	assert(mInput->Data.mAllocSize == mOutput->Data.mAllocSize);

	return true;
}

void LeakyReLUNeuron::forward()
{
	for (uint64_t i = 0; i < mInput->Data.mAllocSize; i++)
	{
		mOutput->Data(i) = std::max(LeakFactor*mInput->Data(i), mInput->Data(i));
	}
	// printf("%f %f %f\n", mOutput->Data(0), mInput->Data(0), LeakFactor*mInput->Data(0));
}

void LeakyReLUNeuron::backprop()
{
	for (uint64_t i = 0; i < mInput->Data.mAllocSize; i++)
	{
		mInput->Delta(i) += mOutput->Data(i) < 0.0? LeakFactor*mOutput->Delta(i): mOutput->Delta(i);
	}
}

std::vector<Blob*> LeakyReLUNeuron::getVariables()
{
	return std::vector<Blob*>();
}
