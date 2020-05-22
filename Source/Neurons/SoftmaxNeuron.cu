#include "Neurons/SoftmaxNeuron.h"

__global__ void softmax_forward(int size, float *input_data, float* output_data)
{
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < size)
	{
		output_data[tid] = 1.0/(1.0+exp(-input_data[tid]));
	}
}

__global__ void softmax_backprop(int size, float *input_data, float *input_delta, float* output_data, float* output_delta)
{
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < size)
	{
		input_delta[tid] += (1.0 - output_data[tid])*output_data[tid];
	}
}

void SoftmaxNeuron::forwardGPU()
{
	// int N = mInput->Data.mAllocSize;
	// int NUM_THREADS = 1 << 10;
	// int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
	// softmax_forward<<<NUM_BLOCKS,NUM_THREADS>>>(mInput->Data.mAllocSize,
	// 											mInput->Data.mDataGPU, mOutput->Data.mDataGPU);

	float alpha = 1.0f;
	float beta = 0.0f;
	checkCUDNN(cudnnSoftmaxForward(gCudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, mInputDesc, mInput->Data.mDataGPU, &beta, mOutputDesc, mOutput->Data.mDataGPU));
}

void SoftmaxNeuron::backpropGPU()
{
	// int N = mInput->Data.mAllocSize;
	// int NUM_THREADS = 1 << 10;
	// int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
	// softmax_backprop<<<NUM_BLOCKS,NUM_THREADS>>>(mInput->Data.mAllocSize,
	// 											mInput->Data.mDataGPU, mInput->Delta.mDataGPU,
	// 											mOutput->Data.mDataGPU, mOutput->Delta.mDataGPU);

	float alpha = 1.0f;
	float beta = 1.0f;
	cudnnSoftmaxBackward(gCudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, mOutputDesc, mOutput->Data.mDataGPU, mOutputDesc, mOutput->Delta.mDataGPU, &beta,
						 mInputDesc, mInput->Delta.mDataGPU);
}