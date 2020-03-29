#include "Neurons/LeakyReLUNeuron.h"

__global__ void leaky_relu_forward(int size, float leak_factor, float *input_data, float* output_data)
{
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < size)
	{
		output_data[tid] = leak_factor * input_data[tid] > input_data[tid] ? leak_factor * input_data[tid] : input_data[tid];
		// if(tid==0)
		// {
		// 	printf("setting %d to %f %f, input %f\n", tid, output_data[tid],leak_factor,input_data[tid]);
		// 	printf("locatins %d %d\n", output_data, input_data);
		// }
	}
}
__global__ void leaky_relu_backprop(int size, float leak_factor, float *input_data, float *input_delta, float* output_data, float* output_delta)
{
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < size)
	{
		input_delta[tid] += output_data[tid] < 0.0f ? leak_factor * output_delta[tid] : output_delta[tid];
	}
}

__global__ void check(float* ptr,float val)
{
	if (ptr[0] <= val+0.01 && ptr[0]>=val-0.01)
	{
		printf("check passed %f %f %d\n",ptr[0],val,ptr);
	}
	else
	{
		printf("check failed %f %f %d\n",ptr[0],val,ptr);
	}
}

void LeakyReLUNeuron::forwardGPU()
{
	int N = mInput->Data.mAllocSize;
	int NUM_THREADS = 1 << 10;
	int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
	// printGPU<<<1, 1>>>(mInput->Data.mAllocSize,1,mInput->Data.mAllocSize,mInput->Data.mDataGPU);
	// gpuErrChk(cudaDeviceSynchronize());
	gpuErrChk(cudaDeviceSynchronize());
	leaky_relu_forward<<<NUM_BLOCKS,NUM_THREADS>>>(mInput->Data.mAllocSize, LeakFactor,
													   mInput->Data.mDataGPU, mOutput->Data.mDataGPU);
	// printf("%s gpu data: %d %d \n", Name.c_str(), mInput->Data.mDataGPU, mOutput->Data.mDataGPU);
	forwardCPU();
	gpuErrChk(cudaDeviceSynchronize());
	// check<<<1, 1>>>(mOutput->Data.mDataGPU, mOutput->Data.mData[0]);
	// check<<<1, 1>>>(mInput->Data.mDataGPU, mInput->Data.mData[0]);
	gpuErrChk(cudaDeviceSynchronize());

}

void LeakyReLUNeuron::backpropGPU()
{
	int N = mInput->Data.mAllocSize;
	int NUM_THREADS = 1 << 10;
	int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
	leaky_relu_backprop<<<NUM_BLOCKS,NUM_THREADS>>>(mInput->Data.mAllocSize, LeakFactor,
												mInput->Data.mDataGPU, mInput->Delta.mDataGPU,
												mOutput->Data.mDataGPU, mOutput->Delta.mDataGPU);
}