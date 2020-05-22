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
	Float error = calculateErrorGPU();
	// Float cpuerror = calculateErrorCPU();
	// printf("error %f %f\n", error, cpuerror);
	// assert(error <= cpuerror+10 && error >=cpuerror-10);
#else
	Float error = calculateErrorCPU();
#endif
	// printf("err %d %f %f %f\n", 0, mOutput->Delta(0), mOutput->Data(0), mTarget(0));
	return error;
}

Float MeanSquaredError::calculateErrorCPU()
{
	if (mTarget.mData == NULL)
		return 0;
	
	Float error = 0.0f;
	for (uint64_t i = 0; i < mOutput->Data.mAllocSize; i++)
	{
		error += 0.5*(mOutput->Data(i) - mTarget(i))*(mOutput->Data(i) - mTarget(i));
		mOutput->Delta(i) += mOutput->Data(i) - mTarget(i);
	}
	return error;
}

void MeanSquaredError::backprop()
{
}
