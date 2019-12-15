#include "Neurons/DiagNeuron.h"

DiagNeuron::DiagNeuron()
{
}

DiagNeuron::DiagNeuron(Blob* input, Blob* output, Tensor pad_value)
	: mInput(input), mOutput(output), PadValue(pad_value)
{
	assert(input->Data.mShape.size() == 4);

	BatchSize = input->Data.mShape[0];
	InputHeight = input->Data.mShape[1];
	InputWidth = input->Data.mShape[2];
	InputDepth = input->Data.mShape[3];

	assert(output->Data.mShape.size() == 2);
	OutputCols = output->Data.mShape[1];
	assert(OutputCols == InputHeight*InputDepth);
	OutputRows = output->Data.mShape[0];
	DiagCount = (InputHeight + InputWidth - 1) * 2;
	assert(OutputRows == BatchSize*DiagCount);

	assert(pad_value.mSize == InputDepth);
}

DiagNeuron::~DiagNeuron()
{
	PadValue.freemem();
}

void DiagNeuron::forward()
{
	for (uint64_t batch = 0; batch < BatchSize; batch++)
	{
		uint64_t sub_batch = 0;
		for (uint64_t y = 0; y < InputHeight; y++)
		{
			uint64_t id = 0;
			for (uint64_t t = 0; t < InputWidth-y; t++)
			{
				memcpy(&mOutput->Data(batch*DiagCount + sub_batch, id), &mInput->Data(batch, y + t, t, 0), sizeof(Float)*InputDepth);
				id += InputDepth;
			}
			for (uint64_t t = InputWidth - y; t < InputWidth; t++)
			{
				memcpy(&mOutput->Data(batch*DiagCount + sub_batch, id), PadValue.mData, sizeof(Float)*InputDepth);
				id += InputDepth;
			}
			sub_batch++;
		}

		for (uint64_t y = 0; y < InputHeight; y++)
		{
			uint64_t id = 0;
			for (uint64_t t = 0; t < InputWidth-y; t++)
			{
				memcpy(&mOutput->Data(batch*DiagCount + sub_batch, id), &mInput->Data(batch, y + t, InputWidth-t-1, 0), sizeof(Float)*InputDepth);
				id += InputDepth;
			}
			for (uint64_t t = InputWidth - y; t < InputWidth; t++)
			{
				memcpy(&mOutput->Data(batch*DiagCount + sub_batch, id), PadValue.mData, sizeof(Float)*InputDepth);
				id += InputDepth;
			}
			sub_batch++;
		}

		for (uint64_t y = 1; y < InputHeight; y++)
		{
			uint64_t id = 0;
			for (uint64_t t = 0; t < InputWidth-y; t++)
			{
				memcpy(&mOutput->Data(batch*DiagCount + sub_batch, id), &mInput->Data(batch, t, y + t, 0), sizeof(Float)*InputDepth);
				id += InputDepth;
			}
			for (uint64_t t = InputWidth - y; t < InputWidth; t++)
			{
				memcpy(&mOutput->Data(batch*DiagCount + sub_batch, id), PadValue.mData, sizeof(Float)*InputDepth);
				id += InputDepth;
			}
			sub_batch++;
		}

		for (uint64_t y = 1; y < InputHeight; y++)
		{
			uint64_t id = 0;
			for (uint64_t t = 0; t < InputWidth-y; t++)
			{
				memcpy(&mOutput->Data(batch*DiagCount + sub_batch, id), &mInput->Data(batch, t, InputWidth - y - t - 1, 0), sizeof(Float)*InputDepth);
				id += InputDepth;
			}
			for (uint64_t t = InputWidth - y; t < InputWidth; t++)
			{
				memcpy(&mOutput->Data(batch*DiagCount + sub_batch, id), PadValue.mData, sizeof(Float)*InputDepth);
				id += InputDepth;
			}
			sub_batch++;
		}
	}
}

void DiagNeuron::backprop()
{
}

std::vector<Blob*> DiagNeuron::getVariables()
{
	return std::vector<Blob*>();
}
