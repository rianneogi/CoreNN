#include "Neurons/FileNeuron.h"

FileNeuron::FileNeuron()
{
}

FileNeuron::FileNeuron(Blob* input, Blob* output) : mInput(input), mOutput(output)
{
	BatchSize = input->Data.mShape[0];

	assert(input->Data.mShape.size() == 4);

	InputHeight = input->Data.mShape[1];
	InputWidth = input->Data.mShape[2];
	InputDepth = input->Data.mShape[3];

	assert(output->Data.mShape.size() == 2);
	OutputCols = output->Data.mShape[1];
	assert(OutputCols == InputHeight*InputDepth);
	OutputRows = output->Data.mShape[0];
	assert(OutputRows == BatchSize*InputWidth);
}

FileNeuron::~FileNeuron()
{
}

void FileNeuron::forward()
{
	uint64_t id = 0;
	for (uint64_t batch = 0; batch < BatchSize; batch++)
	{
		for (uint64_t x = 0; x < InputWidth; x++)
		{
			for (uint64_t y = 0; y < InputHeight; y++)
			{
				memcpy(&mOutput->Data(id, y*InputDepth), &mInput->Data(batch, y, x, 0), InputDepth * sizeof(Float));
			}
			id++;
		}
	}
}

void FileNeuron::backprop()
{
}

std::vector<Blob*> FileNeuron::getVariables()
{
	return std::vector<Blob*>();
}
