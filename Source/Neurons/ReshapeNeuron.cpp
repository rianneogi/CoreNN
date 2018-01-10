#include "Neurons/ReshapeNeuron.h"

ReshapeNeuron::ReshapeNeuron()
{
}

ReshapeNeuron::ReshapeNeuron(Blob* input, Blob* output, TensorShape output_shape) : mInput(input), mOutput(output)
{
	assert(input->Data.mSize == output->Data.mSize);
	InputShape = input->Data.mShape;
	OutputShape = output_shape;
}

ReshapeNeuron::~ReshapeNeuron()
{
}

void ReshapeNeuron::forward()
{
	mOutput->reshape(OutputShape);
	//memcpy(&mOutput->Data, &mInput->Data, sizeof(Float)*mInput->Data.mSize);
	/*for (int i = 0; i < mOutput->Data.mSize; i++)
	{
		mOutput->Data(i) = mInput->Data(i);
	}*/
}

void ReshapeNeuron::backprop()
{
	mOutput->reshape(InputShape);
	//memcpy(&mInput->Delta, &mOutput->Delta, sizeof(Float)*mInput->Delta.mSize);
	/*for (int i = 0; i < mInput->Delta.mSize; i++)
	{
		mInput->Delta(i) = mOutput->Delta(i);
	}*/
}

std::vector<Blob*> ReshapeNeuron::getVariables()
{
	return std::vector<Blob*>();
}
