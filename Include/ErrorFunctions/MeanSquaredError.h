#pragma once

#include "../ErrorFunction.h"

class MeanSquaredError : public ErrorFunction
{
public:
	MeanSquaredError();
	MeanSquaredError(Blob* output);
	MeanSquaredError(Blob* output, Tensor target);
	~MeanSquaredError();

	Float calculateError();
	void backprop();
};

