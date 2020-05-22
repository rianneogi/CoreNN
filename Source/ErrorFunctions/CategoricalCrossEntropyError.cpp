#include "ErrorFunctions/CategoricalCrossEntropyError.h"

float epsilon = 0.00005;

CategoricalCrossEntropyError::~CategoricalCrossEntropyError()
{
}

Float CategoricalCrossEntropyError::calculateError()
{
#ifdef USE_GPU
	float f = calculateErrorGPU();
	// float f2 = calculateErrorCPU();
	// printf("err %f %f\n", f, f2);
	// assert(f >= f2 - 0.5 && f <= f2 + 0.5);
	return f;
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
		error += -mTarget(i) * log(mOutput->Data(i)+epsilon);
		// error += -mTarget(i) * mOutput->Data(i);
		mOutput->Delta(i) += -mTarget(i) / mOutput->Data(i);
	}

	return error;
}

void CategoricalCrossEntropyError::backprop()
{
}
