#include "Neurons/ReshapeNeuron.h"

ReshapeNeuron::ReshapeNeuron()
{
}

ReshapeNeuron::ReshapeNeuron(Blob* input, TensorShape output_shape) : mInput(input)
{
	//assert(input->Data.mSize == output->Data.mSize);
	InputShape = input->Data.mShape;
	InputOffset = input->Data.mOffset;
	InputSubshape = input->Data.mShape;
	
	OutputShape = output_shape;
	OutputOffset = std::vector<uint64_t>(4,0);
	OutputSubshape = output_shape;
}

ReshapeNeuron::ReshapeNeuron(Blob* input, TensorShape output_shape, TensorShape output_offset, TensorShape output_subshape) : mInput(input)
{
	// assert(input->Data.mSize == output->Data.mSize);
	InputShape = input->Data.mAllocShape;
	InputOffset = input->Data.mOffset;
	InputSubshape = input->Data.mShape;
	
	OutputShape = output_shape;
	OutputOffset = output_offset;
	OutputSubshape = output_subshape;
}

ReshapeNeuron::~ReshapeNeuron()
{
}

bool ReshapeNeuron::init()
{
	uint64_t size = 1;
	for(size_t i = 0;i<OutputShape.size();i++)
	{
		size *= OutputShape[i];
	}
	assert(size==mInput->Data.mAllocSize);
	forward();
	
	return true;
}

void ReshapeNeuron::forward()
{
	mInput->reshape(OutputShape, OutputOffset, OutputSubshape);
	//memcpy(&mOutput->Data, &mInput->Data, sizeof(Float)*mInput->Data.mSize);
	/*for (int i = 0; i < mOutput->Data.mSize; i++)
	{
		mOutput->Data(i) = mInput->Data(i);
	}*/
}

void ReshapeNeuron::backprop()
{
	mInput->reshape(InputShape, InputOffset, InputSubshape);
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

void ReshapeNeuron::reset()
{
	mInput->reshape(InputShape, InputOffset, InputSubshape);
}
