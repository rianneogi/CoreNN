#include "Neurons/TanhNeuron.h"

#warning todo: test this

__global__ void tanh_forward(int size, float *input_data, float* output_data)
{
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < size)
	{
		output_data[tid] = tanh(input_data[tid]);
	}
}
__global__ void tanh_backprop(int size, float *input_data, float *input_delta, float* output_data, float* output_delta)
{
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < size)
	{
		input_delta[tid] += output_delta[tid]*(1.0 - output_data[tid]*output_data[tid]);
	}
}

void TanhNeuron::forwardGPU()
{
	int N = mInput->Data.mAllocSize;
	int NUM_THREADS = 1 << 10;
	int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
	// printGPU<<<1, 1>>>(mInput->Data.mAllocSize,1,mInput->Data.mAllocSize,mInput->Data.mDataGPU);
	// gpuErrChk(cudaDeviceSynchronize());
	// gpuErrChk(cudaDeviceSynchronize());
	tanh_forward<<<NUM_BLOCKS,NUM_THREADS>>>(mInput->Data.mAllocSize,
													   mInput->Data.mDataGPU, mOutput->Data.mDataGPU);
	// printf("%s gpu data: %d %d \n", Name.c_str(), mInput->Data.mDataGPU, mOutput->Data.mDataGPU);
	// forwardCPU();
	// gpuErrChk(cudaDeviceSynchronize());
	// check<<<1, 1>>>(mOutput->Data.mDataGPU, mOutput->Data.mData[0]);
	// check<<<1, 1>>>(mInput->Data.mDataGPU, mInput->Data.mData[0]);
	// gpuErrChk(cudaDeviceSynchronize());

}

void TanhNeuron::backpropGPU()
{
	int N = mInput->Data.mAllocSize;
	int NUM_THREADS = 1 << 10;
	int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
	tanh_backprop<<<NUM_BLOCKS,NUM_THREADS>>>(mInput->Data.mAllocSize,
												mInput->Data.mDataGPU, mInput->Delta.mDataGPU,
												mOutput->Data.mDataGPU, mOutput->Delta.mDataGPU);
}