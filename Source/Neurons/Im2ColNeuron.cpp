#include "Neurons/Im2ColNeuron.h"

Im2ColNeuron::Im2ColNeuron() : Neuron()
{
}

Im2ColNeuron::Im2ColNeuron(Blob* input, Blob* output, uint64_t field_width, uint64_t field_height)
	: mInput(input), mOutput(output), FieldWidth(field_width), FieldHeight(field_height)
{
}

Im2ColNeuron::~Im2ColNeuron()
{
}

bool Im2ColNeuron::init()
{
	BatchSize = mInput->Data.mShape[0];

	assert(mInput->Data.mShape.size() == 4);

	InputHeight = mInput->Data.mShape[1];
	InputWidth = mInput->Data.mShape[2];
	InputDepth = mInput->Data.mShape[3];

	assert(mOutput->Data.mShape.size() == 2);
	OutputCols = mOutput->Data.mShape[1];
	assert(OutputCols == FieldHeight*FieldWidth*InputDepth);
	OutputRows = mOutput->Data.mShape[0];
	FieldCount = (InputWidth - FieldWidth + 1)*(InputHeight - FieldHeight + 1);
	assert(OutputRows == BatchSize*FieldCount);

	assert(FieldHeight % 2 == 1 && FieldWidth % 2 == 1 && "Only odd sized receptive fields");

	return true;
}

void Im2ColNeuron::forward()
{
	for (uint64_t batch = 0;batch<BatchSize;batch++)
	{
		uint64_t sub_batch = 0;
		for (uint64_t y = 0;y <= InputHeight - FieldHeight;y++)
		{
			for (uint64_t x = 0;x <= InputWidth - FieldWidth;x++)
			{
				uint64_t id = 0;
				for(uint64_t i = y;i < y+FieldHeight;i++)
				{
					assert(mOutput->Data.mData != 0 && mInput->Data.mData!=0);
					assert(mOutput->Data.mData==mOutput->Data.mStart);
					assert(mInput->Data.mData==mInput->Data.mStart);
					printf("data %d %d \n",mInput->Data.mData,mOutput->Data.mData);
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
	for (uint64_t batch = 0;batch<BatchSize;batch++)
	{
		uint64_t sub_batch = 0;
		for (uint64_t y = 0;y <= InputHeight - FieldHeight;y++)
		{
			for (uint64_t x = 0;x <= InputWidth - FieldWidth;x++)
			{
				uint64_t id = 0;
				for(uint64_t i = y;i < y+FieldHeight;i++)
				{
					for(uint64_t j = x;j < x+FieldWidth;j++)
					{
						for(uint64_t k = 0;k<InputDepth;k++) 
						{
							mInput->Delta(batch, i, j, k) += mOutput->Delta(batch*FieldCount + sub_batch, id);
							id++;
						}
					}
				}
				sub_batch++;
			}
		}
	}
}

std::vector<Blob*> Im2ColNeuron::getVariables()
{
	return std::vector<Blob*>();
}
