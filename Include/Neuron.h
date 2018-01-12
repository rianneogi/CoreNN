#pragma once

#include "Initializer.h"

class Neuron
{
public:
	Neuron();
	virtual ~Neuron();

	virtual void forward() = 0;
	virtual void backprop() = 0;

	virtual std::vector<Blob*> getVariables();
	virtual void reset();
	virtual bool init();
};
