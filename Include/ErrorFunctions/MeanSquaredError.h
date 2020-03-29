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
	Float calculateErrorGPU();
	Float calculateErrorCPU();
	void backprop();
};

float mse_calculate(int size, float *target, float *output_data, float *output_delta);

