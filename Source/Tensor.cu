#include "Tensor.h"

void Tensor::allocateGPU()
{
	// if (mMemory != NULL)
	// {
	// 	freeGPU();
	// }
	// cl_int err;
	// mMemory = clCreateBuffer(gCLContext, CL_MEM_READ_WRITE, mSize * sizeof(cl_float), NULL, &err);
	// if (err != CL_SUCCESS)
	// {
	// 	printf("ERROR: allocating tensor GPU: %d\n", err);
	// }
#ifdef USE_GPU
	if(mDataGPU!=NULL)
	{
		freeGPU();
	}
	gpuErrChk(cudaMalloc(&mDataGPU, mAllocSize * sizeof(Float)));
#endif
}


void Tensor::freeGPU()
{
	// if (mMemory != NULL)
	// {
	// 	clReleaseMemObject(mMemory);
	// 	mMemory = NULL;
	// }
#ifdef USE_GPU
	if(mDataGPU!=NULL)
	{
		gpuErrChk(cudaFree(mDataGPU));
		mDataGPU = NULL;
	}
#endif
}

void Tensor::copyToGPU()
{
#ifdef USE_GPU
	gpuErrChk(cudaMemcpy(mDataGPU, mData, mAllocSize * sizeof(Float), cudaMemcpyHostToDevice));
#endif
	// if (mMemory != NULL && mData != NULL)
	// {
	// 	cl_int err = clEnqueueWriteBuffer(gCLQueue, mMemory, CL_TRUE, 0, mSize * sizeof(cl_float), mData, 0, NULL, NULL);
	// 	if (err != CL_SUCCESS)
	// 	{
	// 		printf("ERROR: copytoGPU: %d\n", err);
	// 	}
	// }



	// /*cl_int err = clblasWriteMatrix(clblasRowMajor, mSize * sizeof(cl_float), mSize * sizeof(cl_float), sizeof(cl_float),
	// mData, 0, cols(), mMemory, 0, cols(),
	// gCLQueue, 1, NULL);
	// if (err != CL_SUCCESS)
	// {
	// printf("ERROR: copytoGPU: %d\n", err);
	// }*/
}

void Tensor::copyToCPU()
{
#ifdef USE_GPU
	gpuErrChk(cudaMemcpy(mData, mDataGPU, mAllocSize * sizeof(Float), cudaMemcpyDeviceToHost));
#endif
	// if (mMemory != NULL && mData != NULL)
	// {
	// 	cl_int err = clEnqueueReadBuffer(gCLQueue, mMemory, CL_TRUE, 0, mSize * sizeof(cl_float), mData, 0, NULL, NULL);
	// 	if (err != CL_SUCCESS)
	// 	{
	// 		printf("ERROR: copytoCPU: %d\n", err);
	// 	}
	// }
	// /*cl_int err = clblasReadMatrix(clblasRowMajor, mSize * sizeof(cl_float), mSize * sizeof(cl_float), sizeof(cl_float),
	// mMemory, 0, cols(), mData, 0, cols(),
	// gCLQueue, 1, NULL);
	// if (err != CL_SUCCESS)
	// {
	// printf("ERROR: copytoGPU: %d\n", err);
	// }*/
}

// __global__ void Tensor::printGPU() const
// {
// 	for (uint64_t i = 0; i < mShape[0]; i++)
// 	{
// 		for (uint64_t j = 0; j < mShape[1]; j++)
// 		{
// 			printf("%f ", mDataGPU[i*mLD+j]);
// 		}
// 		printf("\n");
// 	}
// }


__global__ void printGPU(int m, int n, int ld, float* data)
{
	for (uint64_t i = 0; i < n; i++)
	{
		for (uint64_t j = 0; j < m; j++)
		{
			printf("%f ", data[j*ld+i]);
		}
		printf("\n");
	}
}