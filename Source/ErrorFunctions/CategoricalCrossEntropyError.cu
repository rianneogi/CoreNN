#include "ErrorFunctions/CategoricalCrossEntropyError.h"

struct CCEFunctor
{
	CCEFunctor () {}

	__host__ __device__ float operator()(const float& x, const float& y) const
	{
		return -x*log(y);
	}
};

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

	thrust::device_vector<float> v(mOutput->Data.mAllocSize);
	// float *v_dptr = thrust::raw_pointer_cast(&v[0]);
	// cudaMemcpy(v_dptr, mOutput->Data.mDataGPU, mOutput->Data.mSize, cudaMemcpyDeviceToDevice);

	thrust::device_ptr<float> data_ptr = thrust::device_pointer_cast(mOutput->Data.mDataGPU);
	thrust::device_ptr<float> target_ptr = thrust::device_pointer_cast(mTarget.mDataGPU);
	thrust::transform(data_ptr, data_ptr + N, target_ptr, v.begin(), CCEFunctor());
	float error = thrust::reduce(v.begin(), v.end());

	cce_backprop<<<NUM_BLOCKS, NUM_THREADS>>>(mOutput->Data.mAllocSize,
											  mOutput->Data.mDataGPU, mOutput->Delta.mDataGPU, mTarget.mDataGPU);

	return error;
}