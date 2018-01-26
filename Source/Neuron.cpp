#include "Neuron.h"

Neuron::Neuron()
{
	// printf("WARNING: default constructor for neuron called\n");
}

Neuron::~Neuron()
{
}

std::vector<Blob*> Neuron::getVariables()
{
    return std::vector<Blob*>();
}

bool Neuron::init()
{
    printf("default init called\n");
    // _getch();
	return true;
}

void Neuron::reset()
{

}
