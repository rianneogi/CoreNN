#pragma once

#include "Initializer.h"

class Neuron
{
public:
	std::string Name;
	
	Neuron();
	virtual ~Neuron();

	virtual void forward() = 0;
	virtual void backprop() = 0;

	virtual std::vector<Blob*> getVariables();
	virtual void reset();
	virtual bool init();
};
