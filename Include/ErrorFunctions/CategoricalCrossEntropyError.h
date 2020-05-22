#pragma once

#include "../ErrorFunction.h"

class CategoricalCrossEntropyError : public ErrorFunction
{
public:
	uint64_t BatchSize;
	uint64_t NumCategories;
	thrust::device_vector<float> intermediateVector; //vector to store an intermediate computation

	CategoricalCrossEntropyError();
	CategoricalCrossEntropyError(Blob* output);
	CategoricalCrossEntropyError(Blob* output, Tensor target);
	~CategoricalCrossEntropyError();

	Float calculateError();
	Float calculateErrorGPU();
	Float calculateErrorCPU();
	void backprop();
};
