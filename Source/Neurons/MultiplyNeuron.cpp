#include "Neurons/MultiplyNeuron.h"

MultiplyNeuron::MultiplyNeuron()
{
}

MultiplyNeuron::MultiplyNeuron(Blob* input1, Blob* input2, Blob* output) : Input1(input1), Input2(input2), Output(output)
{
}

MultiplyNeuron::~MultiplyNeuron()
{
}

void MultiplyNeuron::forward()
{
	for (uint64_t i = 0; i < Input1->Data.mSize; i++)
	{
		Output->Data(i) = Input1->Data(i) * Input2->Data(i);
	}
}

void MultiplyNeuron::backprop()
{
	for (uint64_t i = 0; i < Input1->Data.mSize; i++)
	{
		Input1->Delta(i) = Output->Delta(i)*Input2->Data(i);
		Input2->Delta(i) = Output->Delta(i)*Input1->Data(i);
	}
}

std::vector<Blob*> MultiplyNeuron::getVariables()
{
	return std::vector<Blob*>();
}
