#pragma once

#include "TensorShape.h"

// extern cl_context gCLContext;
// extern cl_command_queue gCLQueue;

class Tensor
{
public:
	TensorShape mShape;
	uint64_t mSize;
	Float* mData;
	// cl_mem mMemory;
	
	TensorShape mOffset;
	Float* mStart;
	TensorShape mAllocShape;
	uint64_t mAllocSize;
	uint64_t mLD; //Size of leading dimension

	float* mDataGPU;
	float *mStartGPU;

	Tensor();
	//Tensor(const Tensor& other);
	Tensor(const TensorShape& shape); //initialize tensor allocated with given shape
	Tensor(Float* data, const TensorShape& shape, bool is_gpu); //initialize tensor pointing to existing data
	// Tensor(Float* data, const TensorShape& shape, uint64_t ld); //initialize tensor pointing to existing data and specify leading dimension
	Tensor(Float* data, const TensorShape& shape, const TensorShape& offset, const TensorShape& subshape, bool is_gpu); //initialize tensor pointing to existing data and specify leading dimension
	~Tensor();
	
	Float& at(uint64_t a) const; //slow

	Float& operator()(uint64_t a) const; //Raw access from allocated memory (does not use offsets or shape)
	Float& operator()(uint64_t a, uint64_t b) const;
	Float& operator()(uint64_t a, uint64_t b, uint64_t c) const;
	Float& operator()(uint64_t a, uint64_t b, uint64_t c, uint64_t d) const;
	
	void copyFromTensor(const Tensor& other); //dupliates data from other mData with shape mAllocShape
	void copyFromSubtensor(const Tensor& other); //duplicates data from mStart with shape mShape

	void allocateCPU();
	void allocateGPU();

	void freemem();
	void freeCPU();
	void freeGPU();

	void copyToGPU();
	void copyToCPU();

	void setzero();
	void setconstant(Float c);
	void setidentity();

	void reshape(const TensorShape& shape); //Reshapes the allocated shape, and sets subshape to allocated shape
	void reshape(const TensorShape& shape, const TensorShape& offset, const TensorShape& subshape);
	
	Float sum();

	Tensor subtensor(const TensorShape& begin, const TensorShape& size);
	Tensor cut(uint64_t begin, uint64_t len) const; //cuts the tensor based on primary dimension
	Tensor cut2(uint64_t begin, uint64_t len) const; //cuts the tensor based on last dimension
	Tensor submatrix(uint64_t begin_row, uint64_t begin_col, uint64_t rows, uint64_t cols) const;

	Tensor subtensorGPU(const TensorShape& begin, const TensorShape& size);
	Tensor cutGPU(uint64_t begin, uint64_t len) const; //cuts the tensor based on primary dimension
	Tensor cut2GPU(uint64_t begin, uint64_t len) const; //cuts the tensor based on last dimension
	Tensor submatrixGPU(uint64_t begin_row, uint64_t begin_col, uint64_t rows, uint64_t cols) const;

	uint64_t rows() const;
	uint64_t cols() const;

	void print() const; //prints in matrix form assuming raw data is column major
	// __global__  void printGPU() const;
	void print_raw() const;
};

__global__ void printVal(float *ptr);
__global__ void printGPU(int m, int n, int ld, float *data); //prints GPU data as an mxn matrix, assumes column major

inline void gemm_cpu(Tensor* m1, Tensor* m2, Tensor* res, CBLAS_TRANSPOSE trans_m1, CBLAS_TRANSPOSE trans_m2, Float alpha, Float beta)
{
#ifdef NN_DEBUG
	uint64_t M = trans_m1 == CblasNoTrans ? m1->rows() : m1->cols();
	uint64_t N = trans_m2 == CblasNoTrans ? m2->cols() : m2->rows();
	uint64_t K = trans_m1 == CblasNoTrans ? m1->cols() : m1->rows();
	uint64_t L = trans_m2 == CblasNoTrans ? m2->rows() : m2->cols();
	assert(K == L);
	assert(M == res->rows());
	assert(N == res->cols());
#endif
	// printf("%d %d %d %d %d %d %d %d %d\n", m1->mData, m1->mStart, m2->mData, m2->mStart, res->mData, res->mStart, m1->mLD, m2->mLD, res->mLD);
	// #warning remove asserts when fixed
	// assert(m1->mData==m1->mStart && m2->mData==m2->mStart && res->mData == res->mStart);
	// assert(m1->mLD == m1->mAllocShape[1] && m2->mLD == m2->mAllocShape[1] && res->mLD == res->mAllocShape[1]);
	cblas_sgemm(CblasColMajor, trans_m1, trans_m2,
		res->rows(), //M
		res->cols(), //N
		trans_m1 == CblasNoTrans ? m1->cols() : m1->rows(), //K
		alpha,
		m1->mStart, m1->mLD,
		m2->mStart, m2->mLD,
		beta,
		res->mStart, res->mLD);
}

inline void gemm_gpu(Tensor* m1, Tensor* m2, Tensor* res, cublasOperation_t trans_m1, cublasOperation_t trans_m2, Float alpha, Float beta)
{
#ifdef NN_DEBUG
	uint64_t M = trans_m1 == CUBLAS_OP_N ? m1->rows() : m1->cols();
	uint64_t N = trans_m2 == CUBLAS_OP_N ? m2->cols() : m2->rows();
	uint64_t K = trans_m1 == CUBLAS_OP_N ? m1->cols() : m1->rows();
	uint64_t L = trans_m2 == CUBLAS_OP_N ? m2->rows() : m2->cols();
	assert(K == L);
	assert(M == res->rows());
	assert(N == res->cols());
#endif
	// #warning todo: test	
	auto err = cublasSgemm_v2(gCuHandle, trans_m1, trans_m2,
				   res->rows(),										  //M
				   res->cols(),										  //N
				   trans_m1 == CUBLAS_OP_N ? m1->cols() : m1->rows(), //K
				   &alpha,
				   m1->mDataGPU, m1->mLD,
				   m2->mDataGPU, m2->mLD,
				   &beta,
				   res->mDataGPU, res->mLD);
	if(err!=CUBLAS_STATUS_SUCCESS)
	{
		printf("ERROR during sgemm %d\n", err);
		assert(false);
		// 13 = CUBLAS_STATUS_EXECUTION_FAILED
		// 14 = CUBLAS_STATUS_INTERNAL_ERROR
	}
	// printf("%d\n", err);
}

inline void add_vectors(Tensor* src, Tensor* dest, Float alpha) //test this
{
	cblas_saxpy(src->mSize, alpha, src->mData, 1, dest->mData, 1);
}

inline void saxpy_gpu(Tensor* src, Tensor* dest, Float alpha, int xinc, int yinc) //test this
{
	cublasSaxpy_v2(gCuHandle,dest->mSize, &alpha, src->mStartGPU, xinc, dest->mStartGPU, yinc);
}
