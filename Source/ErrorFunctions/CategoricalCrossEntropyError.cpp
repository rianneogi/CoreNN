#include "ErrorFunctions/CategoricalCrossEntropyError.h"

CategoricalCrossEntropyError::CategoricalCrossEntropyError() : ErrorFunction()
{
}

CategoricalCrossEntropyError::CategoricalCrossEntropyError(Blob* output) : ErrorFunction(output, Tensor(output->Data.mShape))
{
}

CategoricalCrossEntropyError::CategoricalCrossEntropyError(Blob* output, Tensor target) : ErrorFunction(output, target)
{
	BatchSize = output->Data.mShape[0];
	NumCategories = output->Data.mSize / BatchSize;
}

CategoricalCrossEntropyError::~CategoricalCrossEntropyError()
{
}

Float CategoricalCrossEntropyError::calculateError()
{
#ifdef USE_GPU
	return calculateErrorGPU();
#elif
	return calculateErrorCPU();
#endif
}

Float CategoricalCrossEntropyError::calculateErrorCPU()
{
	if (mTarget.mData == NULL)
		return 0;

	Float error = 0;
	for (int i = 0; i < mOutput->Data.mSize; i++)
	{
		error += -mTarget(i) * log(mOutput->Data(i));
		mOutput->Delta(i) += -mTarget(i)/mOutput->Data(i);
	}

	return error;
}

void CategoricalCrossEntropyError::backprop()
{
}
