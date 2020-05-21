#include "ErrorFunctions/CategoricalCrossEntropyError.h"

__global__ void cce_backprop(int size, float *output_data, float *output_delta, float* target)
{
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < size)
	{
		output_delta[tid] += -target[tid]/output_data[tid];
	}
}

Float CategoricalCrossEntropyError::calculateErrorGPU()
{
	int N = mOutput->Data.mAllocSize;
	int NUM_THREADS = 1 << 10;
	int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
	cce_backprop<<<NUM_BLOCKS,NUM_THREADS>>>(mOutput->Data.mAllocSize,
											mOutput->Data.mDataGPU, mOutput->Delta.mDataGPU, mTarget.mDataGPU);
}