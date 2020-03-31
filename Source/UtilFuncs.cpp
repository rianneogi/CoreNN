#include "UtilFuncs.h"

cublasHandle_t gCublasHandle;
cudnnHandle_t gCudnnHandle;

double clamp(double x)
{
	return x > 1.0 ? 1.0 : (x < 0.0 ? 0.0 : x);
}

void initCuda()
{
	int id;
	cudaGetDevice(&id);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, id);
	printf("Using GPU: %s\n", prop.name);

	cublasCreate_v2(&gCublasHandle);
	checkCUDNN(cudnnCreate(&gCudnnHandle));
}

void cleanupCuda()
{
	cublasDestroy_v2(gCublasHandle);
	checkCUDNN(cudnnDestroy(gCudnnHandle));
}