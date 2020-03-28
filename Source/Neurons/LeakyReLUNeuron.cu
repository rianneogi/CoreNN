#include "Neurons/LeakyReLUNeuron.h"

__global__ void leaky_relu_forward(int size, float leak_factor, float *input_data, float* output_data)
{
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < size)
	{
		output_data[tid] = leak_factor * input_data[tid] > input_data[tid] ? leak_factor * input_data[tid] : input_data[tid];	}
}
__global__ void leaky_relu_backprop(int size, float leak_factor, float *input_data, float *input_delta, float* output_data, float* output_delta)
{
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < size)
	{
		input_delta[tid] += output_data[tid] < 0.0f ? leak_factor * output_delta[tid] : output_delta[tid];
	}
}

void LeakyReLUNeuron::forwardGPU()
{
	leaky_relu_forward<<<1, mInput->Data.mAllocSize>>>(mInput->Data.mAllocSize, LeakFactor,
													   mInput->Data.mDataGPU, mOutput->Data.mDataGPU);
}

void LeakyReLUNeuron::backpropGPU()
{
	leaky_relu_backprop<<<1, mInput->Data.mAllocSize>>>(mInput->Data.mAllocSize, LeakFactor,
												mInput->Data.mDataGPU, mInput->Delta.mDataGPU,
												mOutput->Data.mDataGPU, mOutput->Delta.mDataGPU);
}