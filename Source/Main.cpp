#include "Tests.h"
// #include "CudaTests.h"

// #define M 6
// #define N 5
// #define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

// static __inline__ void modify (cublasHandle_t handle, float *m, int ldm, int n, int p, int q, float alpha, float beta)
// {
//     cublasSscal (handle, n-q+1, &alpha, &m[IDX2F(p,q,ldm)], ldm);
//     cublasSscal (handle, ldm-p+1, &beta, &m[IDX2F(p,q,ldm)], 1);
// }

// int cublas_test ()
// {
//     cudaError_t cudaStat;    
//     cublasStatus_t stat;
//     cublasHandle_t handle;
//     int i, j;
//     float* devPtrA;
//     float* a = 0;
//     a = (float *)malloc (M * N * sizeof (*a));
//     if (!a) {
//         printf ("host memory allocation failed");
//         return EXIT_FAILURE;
//     }
//     for (j = 1; j <= N; j++) {
//         for (i = 1; i <= M; i++) {
//             a[IDX2F(i,j,M)] = (float)((i-1) * M + j);
//         }
//     }
//     cudaStat = cudaMalloc ((void**)&devPtrA, M*N*sizeof(*a));
//     if (cudaStat != cudaSuccess) {
//         printf ("device memory allocation failed");
//         return EXIT_FAILURE;
//     }
//     stat = cublasCreate(&handle);
//     if (stat != CUBLAS_STATUS_SUCCESS) {
//         printf ("CUBLAS initialization failed\n");
//         return EXIT_FAILURE;
//     }
//     stat = cublasSetMatrix (M, N, sizeof(*a), a, M, devPtrA, M);
//     if (stat != CUBLAS_STATUS_SUCCESS) {
//         printf ("data download failed");
//         cudaFree (devPtrA);
//         cublasDestroy(handle);
//         return EXIT_FAILURE;
//     }
//     modify (handle, devPtrA, M, N, 2, 3, 16.0f, 12.0f);
//     stat = cublasGetMatrix (M, N, sizeof(*a), devPtrA, M, a, M);
//     if (stat != CUBLAS_STATUS_SUCCESS) {
//         printf ("data upload failed");
//         cudaFree (devPtrA);
//         cublasDestroy(handle);        
//         return EXIT_FAILURE;
//     }    
//     cudaFree (devPtrA);
//     cublasDestroy(handle);
//     for (j = 1; j <= N; j++) {
//         for (i = 1; i <= M; i++) {
//             printf ("%7.0f", a[IDX2F(i,j,M)]);
//         }
//         printf ("\n");
//     }
//     free(a);
//     return EXIT_SUCCESS;
// }

// cl_platform_id gCLPlatform = 0;
// cl_device_id gCLDevice = 0;
// cl_context_properties gCLProperties[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
// cl_context gCLContext = 0;
// cl_command_queue gCLQueue = 0;

// cl_program gCLProgram;
// cl_kernel gKernelMatAdd;
// cl_kernel gKernelMatSub;

// #include <CL\cl_ext.h>

// inline void checkErr(cl_int err, const char* name) {
// 	if (err != CL_SUCCESS) {
// 		std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
// 		// _getch();
// 	}
// }

// int initCL()
// {
// 	cl_int err;
// 	cl_mem bufA, bufB, bufC;
// 	cl_event event = NULL;
// 	int ret = 0;
// 	/* Setup OpenCL environment. */
// 	err = clGetPlatformIDs(1, &gCLPlatform, NULL);
// 	if (err != CL_SUCCESS) {
// 		printf("clGetPlatformIDs() failed with %d\n", err);
// 		return 1;
// 	}

// 	err = clGetDeviceIDs(gCLPlatform, CL_DEVICE_TYPE_GPU, 1, &gCLDevice, NULL);
// 	if (err != CL_SUCCESS) {
// 		printf("clGetDeviceIDs() failed with %d\n", err);
// 		return 1;
// 	}
// 	gCLProperties[1] = (cl_context_properties)gCLPlatform;
// 	gCLContext = clCreateContext(gCLProperties, 1, &gCLDevice, NULL, NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("clCreateContext() failed with %d\n", err);
// 		return 1;
// 	}
// 	gCLQueue = clCreateCommandQueue(gCLContext, gCLDevice, 0, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("clCreateCommandQueue() failed with %d\n", err);
// 		clReleaseContext(gCLContext);
// 		return 1;
// 	}
// 	/* Setup clblas. */
// 	err = clblasSetup();
// 	if (err != CL_SUCCESS) {
// 		printf("clblasSetup() failed with %d\n", err);
// 		clReleaseCommandQueue(gCLQueue);
// 		clReleaseContext(gCLContext);
// 		return 1;
// 	}

// 	char* c = new char[100];

// 	clGetDeviceInfo(gCLDevice, CL_DEVICE_NAME, 100, c, NULL);
// 	printf("ext: %s\n", c);

// 	std::ifstream file("Source/Kernels/matrix_funcs.cl");
// 	checkErr(file.is_open() ? CL_SUCCESS : -1, "Kernels/matrix_funcs.cl");
// 	std::string prog(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));

// 	gCLProgram = clCreateProgramWithSource(gCLContext, 1, (const char**)&prog, NULL, &err);
// 	checkErr(err, "clCreateProgramFromSource");
// 	err = clBuildProgram(gCLProgram, 1, &gCLDevice, NULL, NULL, NULL);
// 	checkErr(err, "clBuildProgram");
// 	if (err < 0)
// 	{
// 		size_t log_size;
// 		clGetProgramBuildInfo(gCLProgram, gCLDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

// 		char* program_log = (char*)malloc(log_size + 1);

// 		program_log[log_size] = '\0';

// 		clGetProgramBuildInfo(gCLProgram, gCLDevice, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);

// 		std::cout << "Error::clBuildProgram: " << err << std::endl;
// 		std::cout << program_log << std::endl;
// 		free(program_log);
// 		// _getch();
// 		//exit(0);
// 	}

// 	gKernelMatAdd = clCreateKernel(gCLProgram, "MatAdd", &err);
// 	checkErr(err, "MatAdd");
// 	gKernelMatSub = clCreateKernel(gCLProgram, "MatSub", &err);
// 	checkErr(err, "MatSub");
// 	//cl::Program::Sources source(1, std::make_pair(prog.c_str(), prog.length() + 1));
// 	//cl::Program program(context, source);
// 	//err = program.build(devices, "");
// 	//checkErr(err, "Program::build()");

// 	return ret;
// }

// void cleanupOpenCL()
// {
// 	/* Finalize work with clblas. */
// 	clblasTeardown();
// 	/* Release OpenCL working objects. */
// 	clReleaseCommandQueue(gCLQueue);
// 	clReleaseContext(gCLContext);
// }

int main()
{
	srand(time(0));
	initCuda();
	printf("Cublas initialized\n");
	// initCL();

	Clock timer;
	timer.Start();

	// test_tensor();
	// test_subtensor();
	// test_gemm();
	// test_gemm_subtensor();
	// test_im2col();

	// test_cugemm();
	// cublas_vector_add();
	// test_fc();
	// test_cudnn_conv();
	// test_cudnn_forward();
	// test_cublas_matmul();
	test_conv();
	// test_autoencoder();

	// test_gemm_gpu();

	// cublas_test();

	timer.Stop();
	printf("\nTotal time taken: %d\n", timer.ElapsedMilliseconds());

	//_getch();

	//cleanupOpenCL();
	cleanupCuda();

	return 0;
}
