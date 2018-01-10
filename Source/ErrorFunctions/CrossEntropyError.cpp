#include "ErrorFunctions/CrossEntropyError.h"

CrossEntropyError::CrossEntropyError() : ErrorFunction()
{
}

CrossEntropyError::CrossEntropyError(Blob* output) : ErrorFunction(output, Tensor(output->Data.mShape))
{
}

CrossEntropyError::CrossEntropyError(Blob* output, Tensor target) : ErrorFunction(output, target)
{
}

CrossEntropyError::~CrossEntropyError()
{
}

Float CrossEntropyError::calculateError()
{
	if (mTarget.mData == NULL)
		return 0;

	Float error = 0;
	for (int i = 0; i < mOutput->Data.mSize; i++)
	{
		error += -(mTarget(i)*log(mOutput->Data(i)) + (1.0 - mTarget(i))*log((1.0 - mOutput->Data(i))));
		mOutput->Delta(i) += mTarget(i)*(1.0 / mOutput->Data(i)) - (1.0 - mTarget(i))*(1.0 / (1.0 - mOutput->Data(i)));
	}

	return error;
}

void CrossEntropyError::backprop()
{
}
