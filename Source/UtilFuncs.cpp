#include "UtilFuncs.h"

cublasHandle_t gCuHandle;

double clamp(double x)
{
	return x > 1.0 ? 1.0 : (x < 0.0 ? 0.0 : x);
}

void initCublas()
{
	int id;
	cudaGetDevice(&id);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, id);
	printf("Using GPU: %s\n", prop.name);

	cublasCreate_v2(&gCuHandle);
}

void cleanupCublas()
{
	cublasDestroy_v2(gCuHandle);
}