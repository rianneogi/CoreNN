#include "Neurons/RankNeuron.h"

RankNeuron::RankNeuron()
{
}

RankNeuron::RankNeuron(Blob* input, Blob* output) : mInput(input), mOutput(output)
{
	BatchSize = input->Data.mShape[0];

	assert(input->Data.mShape.size() == 4);

	InputHeight = input->Data.mShape[1];
	InputWidth = input->Data.mShape[2];
	InputDepth = input->Data.mShape[3];

	assert(output->Data.mShape.size() == 2);
	OutputCols = output->Data.mShape[1];
	assert(OutputCols == InputWidth*InputDepth);
	OutputRows = output->Data.mShape[0];
	assert(OutputRows == BatchSize*InputHeight);
}

RankNeuron::~RankNeuron()
{
}

void RankNeuron::forward()
{
}

void RankNeuron::backprop()
{
}

std::vector<Blob*> RankNeuron::getVariables()
{
	return std::vector<Blob*>();
}
