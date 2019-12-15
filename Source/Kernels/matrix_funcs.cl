#pragma OPENCL_EXTENSION cl_khr_byte_addressable_store : enable

void MatAdd(__global int* matA, __global int* matB, __global int* matC)
{
	unsigned int i = get_global_id(0);
	matC[i] = matA[i] + matB[i];
}

void MatSub(__global int* matA, __global int* matB, __global int* matC)
{
	unsigned int i = get_global_id(0);
	matC[i] = matA[i] - matB[i];
}