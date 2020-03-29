#include "ErrorFunctions/MeanSquaredError.h"

float mse_calculate(int size,float* target, float* output_data, float* output_delta)
{
	float alpha = 1.0f;
	cublasSaxpy_v2(gCuHandle, size, &alpha, output_data, 1, target, 1);
	cublasScopy_v2(gCuHandle, size, target, 1, output_delta, 1);
	float error = 0.0f;
	cublasSdot(gCuHandle, size, target, 1, target, 1, &error);
	error *= 0.5f;
	return error;
}

Float MeanSquaredError::calculateErrorGPU()
{
	// gpuErrChk(cudaDeviceSynchronize());
	int size = mOutput->Data.mAllocSize;
	float* output_data = mOutput->Data.mDataGPU;
	float* output_delta = mOutput->Delta.mDataGPU;
	float* target = mTarget.mDataGPU;
	float alpha = -1.0f;
	// printGPU<<<1,1>>>(size, 1, size, output_data);
	// gpuErrChk(cudaDeviceSynchronize());
	cudaMemcpy(output_delta, output_data, size * sizeof(float), cudaMemcpyDeviceToDevice);
	// cublasScopy_v2(gCuHandle, size, output_data, 1, output_delta, 1);
	cublasSaxpy_v2(gCuHandle, size, &alpha, target, 1, output_delta, 1);
	float error = 0.0f;
	cublasSdot(gCuHandle, size, output_delta, 1, output_delta, 1, &error);
	error *= 0.5f;
	// gpuErrChk(cudaDeviceSynchronize());
	return error;
	// Float error = mse_calculate(mOutput->Data.mAllocSize,mTarget.mDataGPU,mOutput->Data.mDataGPU,mOutput->Delta.mDataGPU);
}