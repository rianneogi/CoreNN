#include "Optimizers/AdamOptimizer.h"

Float ADAM_EPSILON = 0.000000001;

__global__ void adam_optimizer(int size, float learning_rate, float* data, float* delta, float* velocity, float* momentum, float beta1, float beta2, float epsilon)
{
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < size)
	{
		velocity[tid] = beta2 * velocity[tid] + (1.0 - beta2) * delta[tid] * delta[tid];
		momentum[tid] = beta1 * momentum[tid] + (1.0 - beta1) * delta[tid];
		data[tid] -= (learning_rate * momentum[tid]) / (sqrt(velocity[tid]) + epsilon);
	}
}

void AdamOptimizer::optimizeGPU()
{
	// gpuErrChk(cudaDeviceSynchronize());
	for (size_t i = 0; i < Variables.size(); i++)
	{
		int N = Variables[i]->Delta.mSize;
		int NUM_THREADS = 1 << 10;
		int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
		adam_optimizer<<<NUM_BLOCKS,NUM_THREADS>>>(Variables[i]->Delta.mSize, LearningRate,
													   Variables[i]->Data.mDataGPU, Variables[i]->Delta.mDataGPU, 
													   Velocity[i].mDataGPU, Momentum[i].mDataGPU,
													   Beta1,Beta2,ADAM_EPSILON);

		
		// gpuErrChk(cudaDeviceSynchronize());
		// printf("opt %d/%d %f ", i, Variables.size(), Variables[i]->Data(0));
		// printVal<<<1, 1>>>(Variables[i]->Data.mDataGPU);
		// gpuErrChk(cudaDeviceSynchronize());
		//update momentum
		/*cblas_dscal(Variables[i]->Delta.mSize, 1 - Beta1, Variables[i]->Delta.mData, 1);
		cblas_daxpy(Variables[i]->Delta.mSize, Beta1, Momentum[i].mData, 1, Variables[i]->Delta.mData, 1);

		for (uint64_t j = 0; j < Variables[i]->Delta.mSize; j++)
		{
			Variables[i]->Data(j) -= LearningRate*Momentum[i](j) / (sqrt(Velocity[i](j)) + EPSILON);
		}*/
	}
}