#pragma once

#include "../ErrorFunction.h"

class CrossEntropyError : public ErrorFunction
{
public:
	CrossEntropyError();
	CrossEntropyError(Blob* output);
	CrossEntropyError(Blob* output, Tensor target);
	~CrossEntropyError();

	Float calculateError();
	void backprop();
};
