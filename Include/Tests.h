#pragma once

#include "NNInclude.h"

extern cl_kernel gKernelMatAdd;
extern cl_kernel gKernelMatSub;

//Runs
void test_fc();
void test_conv();

//CPU
void test_gemm();
void test_gemm_subtensor();
void test_tensor();
void test_im2col();
void test_diag();

//GPU
void test_gemm_gpu();
void test_kernel();
