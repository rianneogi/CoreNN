#pragma once

#include "Initializer.h"

class Neuron
{
public:
	Neuron();
	~Neuron();

	virtual void forward() = 0;
	virtual void backprop() = 0;
	virtual std::vector<Blob*> getVariables() = 0;
};
