#include "Neurons/KnightNeuron.h"

KnightNeuron::KnightNeuron()
{
}

KnightNeuron::KnightNeuron(Blob* input, Blob* output, int64_t field_width, int64_t field_height, Tensor pad_value)
	: mInput(input), mOutput(output), PadValue(pad_value)
{
	assert(input->Data.mShape.size() == 4);

	BatchSize = input->Data.mShape[0];
	InputHeight = input->Data.mShape[1];
	InputWidth = input->Data.mShape[2];
	InputDepth = input->Data.mShape[3];

	assert(output->Data.mShape.size() == 2);
	OutputCols = output->Data.mShape[1];
	assert(OutputCols == 9*InputDepth);
	OutputRows = output->Data.mShape[0];
	FieldCount = InputWidth*InputHeight;
	assert(OutputRows == BatchSize*FieldCount);

	assert(pad_value.mSize == InputDepth);
}

KnightNeuron::~KnightNeuron()
{
	PadValue.freemem();
}

void KnightNeuron::forward()
{
	for (int64_t batch = 0; batch < BatchSize; batch++)
	{
		int64_t sub_batch = 0;
		for (int64_t y = 0; y < InputHeight; y++)
		{
			for (int64_t x = 0; x < InputWidth; x++)
			{

			}
		}
	}
}

void KnightNeuron::backprop()
{
}

std::vector<Blob*> KnightNeuron::getVariables()
{
	return std::vector<Blob*>();
}
