#include "Neuron.h"

Neuron::Neuron()
{
	printf("WARNING: default constructor for neuron called\n");
}

Neuron::~Neuron()
{
}

std::vector<Blob*> Neuron::getVariables()
{
    return std::vector<Blob*>();
}

void reset()
{
    
}
