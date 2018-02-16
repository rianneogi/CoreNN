#include "ErrorFunctions/MeanSquaredError.h"

MeanSquaredError::MeanSquaredError() : ErrorFunction()
{
}

MeanSquaredError::MeanSquaredError(Blob* output) : ErrorFunction(output, Tensor(output->Data.mShape))
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
	if (mTarget.mData == NULL)
		return 0;

	Float error = 0;
	for (uint64_t i = 0; i < mOutput->Data.mAllocSize; i++)
	{
		error += 0.5*(mOutput->Data(i) - mTarget(i))*(mOutput->Data(i) - mTarget(i));
		mOutput->Delta(i) += mOutput->Data(i) - mTarget(i);
	}
	// printf("err %f %f %f\n", mOutput->Delta(0), mOutput->Data(0), mTarget(0));
	return error;
}

void MeanSquaredError::backprop()
{
}
