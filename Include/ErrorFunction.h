#pragma once

#include "Neuron.h"

class ErrorFunction
{
public:
	Blob* mOutput;
	Tensor mTarget;

	ErrorFunction();
	ErrorFunction(Blob* output);
	ErrorFunction(Blob* output, Tensor target);
	~ErrorFunction();

	virtual Float calculateError() = 0;
	virtual void backprop() = 0;
};

