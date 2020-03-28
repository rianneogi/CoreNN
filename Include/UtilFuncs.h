#pragma once

#include "Typedefs.h"
#include "Clock.h"

extern cublasHandle_t gCuHandle;

double clamp(double x);

inline double sigmoid(double x)
{
	return (1.0 / (1.0 + exp(-x)));
}

inline double tanh_NN(double x)
{
	return tanh(x);
}

inline double square(double x)
{
	return x*x;
}

inline double rand_init(Float start, Float end)
{
	return ((rand() % 1024) / 1024.0)*(end-start) + start;
}

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void initCublas();
void cleanupCublas();
