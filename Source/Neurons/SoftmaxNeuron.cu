#include "Neurons/SoftmaxNeuron.h"

void SoftmaxNeuron::forwardGPU()
{
	int N = mInput->Data.mAllocSize;
	int NUM_THREADS = 1 << 10;
	int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
	// printGPU<<<1, 1>>>(mInput->Data.mAllocSize,1,mInput->Data.mAllocSize,mInput->Data.mDataGPU);
	// gpuErrChk(cudaDeviceSynchronize());
	// gpuErrChk(cudaDeviceSynchronize());
	// sigmoid_forward<<<NUM_BLOCKS,NUM_THREADS>>>(mInput->Data.mAllocSize,
	// 												   mInput->Data.mDataGPU, mOutput->Data.mDataGPU);
	// printf("%s gpu data: %d %d \n", Name.c_str(), mInput->Data.mDataGPU, mOutput->Data.mDataGPU);
	// forwardCPU();
	// gpuErrChk(cudaDeviceSynchronize());
	// check<<<1, 1>>>(mOutput->Data.mDataGPU, mOutput->Data.mData[0]);
	// check<<<1, 1>>>(mInput->Data.mDataGPU, mInput->Data.mData[0]);
	// gpuErrChk(cudaDeviceSynchronize());
}

void SoftmaxNeuron::backpropGPU()
{
	int N = mInput->Data.mAllocSize;
	int NUM_THREADS = 1 << 10;
	int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
	// sigmoid_backprop<<<NUM_BLOCKS,NUM_THREADS>>>(mInput->Data.mAllocSize,
	// 											mInput->Data.mDataGPU, mInput->Delta.mDataGPU,
	// 											mOutput->Data.mDataGPU, mOutput->Delta.mDataGPU);
}