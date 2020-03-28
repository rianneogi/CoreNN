#include "ErrorFunctions/MeanSquaredError.h"

float mse_calculate(int size,float* target, float* output_data, float* output_delta)
{
	float alpha = 1.0f;
	cublasSaxpy_v2(gCuHandle, size, &alpha, output_data, 1, target, 1);
	cublasScopy_v2(gCuHandle, size, target, 1, output_delta, 1);
	float error = 0.0f;
	cublasSdot(gCuHandle, size, target, 1, target, 1, &error);
	error *= 0.5;
	return error;
}