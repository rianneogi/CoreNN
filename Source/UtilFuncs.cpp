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

// cuBLAS API errors
std::string cublasGetErrorString(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}