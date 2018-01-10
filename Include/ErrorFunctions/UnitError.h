#pragma once

#include "../ErrorFunction.h"

//Error Function with Delta = 1

class UnitError : public ErrorFunction
{
public:
	UnitError();
	UnitError(Blob* output);
	UnitError(Blob* output, Tensor target);
	~UnitError();

	Float calculateError();
	void backprop();
};