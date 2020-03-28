#include "ErrorFunctions/MeanSquaredError.h"

MeanSquaredError::MeanSquaredError() : ErrorFunction()
{
}

MeanSquaredError::MeanSquaredError(Blob* output) : ErrorFunction(output, Tensor(output->Data.mAllocShape))
{
}

MeanSquaredError::MeanSquaredError(Blob* output, Tensor target) : ErrorFunction(output, target)
{
}

MeanSquaredError::~MeanSquaredError()
{
}

Float MeanSquaredError::calculateError()
{
#ifdef USE_GPU
	int size = mOutput->Data.mAllocSize;
	float* output_data = mOutput->Data.mDataGPU;
	float* output_delta = mOutput->Delta.mDataGPU;
	float* target = mTarget.mDataGPU;
	float alpha = 1.0f;
	cublasSaxpy_v2(gCuHandle, size, &alpha, output_data, 1, target, 1);
	cublasScopy_v2(gCuHandle, size, target, 1, output_delta, 1);
	float error = 0.0f;
	cublasSdot(gCuHandle, size, target, 1, target, 1, &error);
	error *= 0.5;
	// return error;
	// Float error = mse_calculate(mOutput->Data.mAllocSize,mTarget.mDataGPU,mOutput->Data.mDataGPU,mOutput->Delta.mDataGPU);
#else
	if (mTarget.mData == NULL)
		return 0;
	
	Float error = 0;
	for (uint64_t i = 0; i < mOutput->Data.mAllocSize; i++)
	{
		error += 0.5*(mOutput->Data(i) - mTarget(i))*(mOutput->Data(i) - mTarget(i));
		mOutput->Delta(i) += mOutput->Data(i) - mTarget(i);
	}
#endif
	// printf("err %d %f %f %f\n", 0, mOutput->Delta(0), mOutput->Data(0), mTarget(0));
	return error;
}

void MeanSquaredError::backprop()
{
}
