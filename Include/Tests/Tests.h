#pragma once

#include "CudaTests.h"

// extern cl_kernel gKernelMatAdd;
// extern cl_kernel gKernelMatSub;

//Runs
void test_fc();
void test_conv();
void test_autoencoder();

//CPU
void test_gemm();
void test_gemm_subtensor();
void test_tensor();
void test_subtensor();
void test_im2col();
void test_diag();

//GPU
// void test_gemm_gpu();
void test_kernel();

//MKL
void test_mkl();
