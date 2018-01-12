#include "Neurons/Im2ColNeuron.h"

Im2ColNeuron::Im2ColNeuron() : Neuron()
{
}

Im2ColNeuron::Im2ColNeuron(Blob* input, Blob* output, uint64_t field_width, uint64_t field_height)
	: mInput(input), mOutput(output), FieldWidth(field_width), FieldHeight(field_height)
{
	BatchSize = input->Data.mShape[0];

	assert(input->Data.mShape.size() == 4);

	InputHeight = input->Data.mShape[1];
	InputWidth = input->Data.mShape[2];
	InputDepth = input->Data.mShape[3];

	assert(output->Data.mShape.size() == 2);
	OutputCols = output->Data.mShape[1];
	assert(OutputCols == FieldHeight*FieldWidth*InputDepth);
	OutputRows = output->Data.mShape[0];
	FieldCount = (InputWidth - FieldWidth + 1)*(InputHeight - FieldHeight + 1);
	assert(OutputRows == BatchSize*FieldCount);

	assert(FieldHeight % 2 == 1 && FieldWidth % 2 == 1 && "Only odd sized receptive fields");
}

Im2ColNeuron::~Im2ColNeuron()
{
}

void Im2ColNeuron::forward()
{
	for (uint64_t batch = 0;batch<BatchSize;batch++)
	{
		uint64_t sub_batch = 0;
		for(uint64_t y = 0;y <= InputHeight - FieldHeight;y++)
		{
			for (uint64_t x = 0;x <= InputWidth - FieldWidth;x++)
			{
				uint64_t id = 0;
				for(uint64_t i = y;i < y+FieldHeight;i++)
				{
					memcpy(&mOutput->Data(batch*FieldCount + sub_batch, id),
						&mInput->Data(batch, i, x, 0), FieldWidth * InputDepth * sizeof(Float));

					id += FieldWidth*InputDepth;
				}
				sub_batch++;
			}
		}
	}

	//Works only for odd sized receptive fields
	// for (uint64_t batch = 0; batch < BatchSize; batch++)
	// {
	// 	uint64_t sub_batch = 0;
	// 	for (uint64_t y = FieldHeight / 2; y < InputHeight - FieldHeight / 2; y++)
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

void Im2ColNeuron::backprop()
{

}

std::vector<Blob*> Im2ColNeuron::getVariables()
{
	return std::vector<Blob*>();
}
