#include "Neurons/AddNeuron.h"

AddNeuron::AddNeuron()
{
}

AddNeuron::AddNeuron(Blob* input1, Blob* input2, Blob* output) : Input1(input1), Input2(input2), Output(output)
{
}

AddNeuron::~AddNeuron()
{
}

void AddNeuron::forward()
{
	for (uint64_t i = 0; i < Input1->Data.mSize; i++)
	{
		Output->Data(i) = Input1->Data(i) + Input2->Data(i);
	}
}

void AddNeuron::backprop()
{
	for (uint64_t i = 0; i < Input1->Data.mSize; i++)
	{
		Input1->Delta(i) = Output->Delta(i);
		Input2->Delta(i) = Output->Delta(i);
	}
}

std::vector<Blob*> AddNeuron::getVariables()
{
	return std::vector<Blob*>();
}
