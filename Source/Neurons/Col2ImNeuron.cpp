#include "Neurons/Col2ImNeuron.h"

Col2ImNeuron::Col2ImNeuron() : Neuron()
{
}

Col2ImNeuron::Col2ImNeuron(Blob* input, Blob* output, uint64_t field_width, uint64_t field_height)
	: mInput(input), mOutput(output), FieldWidth(field_width), FieldHeight(field_height)
{
	BatchSize = output->Data.mShape[0];

	assert(output->Data.mShape.size() == 4);

	OutputHeight = output->Data.mShape[1];
	OutputWidth = output->Data.mShape[2];
	OutputDepth = output->Data.mShape[3];

	assert(input->Data.mShape.size() == 2);
	InputCols = input->Data.mShape[1];
	assert(InputCols == FieldHeight*FieldWidth*OutputDepth);
	InputRows = output->Data.mShape[0];
	FieldCount = (OutputWidth - FieldWidth + 1)*(OutputHeight - FieldHeight + 1);
	assert(InputRows == BatchSize*FieldCount);

	assert(FieldHeight % 2 == 1 && FieldWidth % 2 == 1 && "Only odd sized receptive fields");
}

Col2ImNeuron::~Col2ImNeuron()
{
}

void Col2ImNeuron::forward()
{
	//Works only for odd sized receptive fields
	// for (uint64_t batch = 0; batch < BatchSize; batch++)
	// {
	// 	uint64_t sub_batch = 0;
	// 	for (uint64_t y = FieldHeight / 2; y < OutputHeight - FieldHeight / 2; y++)
	// 	{
	// 		for (uint64_t x = FieldWidth / 2; x < InputWidth - FieldWidth / 2; x++)
	// 		{
	// 			uint64_t id = 0;
	// 			for (uint64_t i = y - FieldHeight / 2; i <= y + FieldHeight / 2; i++)
	// 			{
	// 				memcpy(&mOutput->Data(batch*FieldCount + sub_batch, id),
	// 					&mInput->Data(batch, i, x - FieldWidth / 2, 0), FieldWidth * InputDepth * sizeof(Float));
    //
	// 				id += FieldWidth*InputDepth;
	// 			}
	// 			sub_batch++;
	// 		}
	// 	}
	// }
}

void Col2ImNeuron::backprop()
{
}

std::vector<Blob*> Col2ImNeuron::getVariables()
{
	return std::vector<Blob*>();
}
