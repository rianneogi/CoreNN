#include "Tensor.h"

#define USE_MALLOC

Tensor::Tensor() : mData(NULL), mSize(0), mMemory(NULL), mLD(0)
{
}

//Tensor::Tensor(const Tensor& other) : mData(other.mData), mShape(other.mShape), mSize(other.mSize), mMemory(other.mMemory), mLD(other.mLD)
//{
//}

Tensor::Tensor(const TensorShape& shape) : mData(NULL), mShape(shape), mSize(1), mMemory(NULL), mLD(mShape[mShape.size() - 1])
{
	assert(shape.size() <= 4 && "Max supported tensor shape is 4");
	for (unsigned int x : mShape)
	{
		mSize *= x;
	}
	//printf("Size : %d\n", mSize);
	allocateCPU();
#ifdef USE_GPU
	allocateGPU();
#endif
}

Tensor::Tensor(Float* data, const TensorShape& shape) : mData(data), mShape(shape), mSize(1), mMemory(NULL), mLD(mShape[mShape.size()-1])
{
	assert(shape.size() <= 4 && "Max supported tensor shape is 4");
	for (unsigned int x : mShape)
	{
		mSize *= x;
	}
}

Tensor::Tensor(Float* data, const TensorShape& shape, uint64_t ld) : mData(data), mShape(shape), mSize(1), mMemory(NULL), mLD(ld)
{
	assert(shape.size() <= 4 && "Max supported tensor shape is 4");
	for (unsigned int x : mShape)
	{
		mSize *= x;
	}
}

Tensor::~Tensor()
{
	//if(mSelfAllocated)
	//	freememory();
}

Float& Tensor::operator()(uint64_t a) const
{
#ifdef NN_DEBUG
	assert(a < mSize);
#endif
	//return mData[a];
	return mData[(a/mShape[mShape.size()-1])*mLD + a%mShape[mShape.size()-1]];
}

Float& Tensor::operator()(uint64_t a, uint64_t b) const
{
#ifdef NN_DEBUG
	assert(mShape.size() >= 1);
	assert(a < mShape[0]);
	assert(b < mShape[1]);
#endif
	return mData[a*mLD + b];
}

Float& Tensor::operator()(uint64_t a, uint64_t b, uint64_t c) const
{
#ifdef NN_DEBUG
	assert(mShape.size() >= 2);
	assert(a < mShape[0]);
	assert(b < mShape[1]);
	assert(c < mShape[2]);
#endif
	return mData[a*mShape[1]*mShape[2] + b*mShape[2] + c];
}

Float& Tensor::operator()(uint64_t a, uint64_t b, uint64_t c, uint64_t d) const
{
#ifdef NN_DEBUG
	assert(mShape.size() >= 3);
	assert(a < mShape[0]);
	assert(b < mShape[1]);
	assert(c < mShape[2]);
	assert(d < mShape[3]);
#endif
	return mData[a*mShape[1]*mShape[2]*mShape[3] + b*mShape[2]*mShape[3] + c*mShape[3] + d];
}

void Tensor::copyFromTensor(const Tensor& other)
{
	assert(mSize == other.mSize);
	mShape = other.mShape;
	mLD = other.mLD;
	std::memcpy(mData, other.mData, other.mSize*sizeof(Float));
}

void Tensor::allocateCPU()
{
	if (mData != NULL)
	{
		freeCPU();
	}
	//printf("Allocation tensor of size: %d\n", mSize);
#ifdef USE_MALLOC
	mData = (Float*)malloc(mSize * sizeof(Float));
	if (mData == NULL)
	{
		printf("ERROR: Cant allocate memory for tensor, Size: %d\n", mSize);
	}
#else
	mData = new Float[mSize];
#endif
}

void Tensor::allocateGPU()
{
	if (mMemory != NULL)
	{
		freeGPU();
	}
	cl_int err;
	mMemory = clCreateBuffer(gCLContext, CL_MEM_READ_WRITE, mSize * sizeof(cl_float), NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("ERROR: allocating tensor GPU: %d\n", err);
	}
}

void Tensor::freemem()
{
	freeCPU();
	freeGPU();
}

void Tensor::freeCPU()
{
	if (mData != NULL)
	{
		//printf("Freeing memory: %d\n", mSize);
#ifdef USE_MALLOC
		free(mData);
		mData = NULL;
#else
		delete[] mData;
		mData = NULL;
#endif
	}
}

void Tensor::freeGPU()
{
	if (mMemory != NULL)
	{
		clReleaseMemObject(mMemory);
		mMemory = NULL;
	}
}


void Tensor::copyToGPU()
{
	if (mMemory != NULL && mData != NULL)
	{
		cl_int err = clEnqueueWriteBuffer(gCLQueue, mMemory, CL_TRUE, 0, mSize * sizeof(cl_float), mData, 0, NULL, NULL);
		if (err != CL_SUCCESS)
		{
			printf("ERROR: copytoGPU: %d\n", err);
		}
	}
	/*cl_int err = clblasWriteMatrix(clblasRowMajor, mSize * sizeof(cl_float), mSize * sizeof(cl_float), sizeof(cl_float),
	mData, 0, cols(), mMemory, 0, cols(),
	gCLQueue, 1, NULL);
	if (err != CL_SUCCESS)
	{
	printf("ERROR: copytoGPU: %d\n", err);
	}*/
}

void Tensor::copyToCPU()
{
	if (mMemory != NULL && mData != NULL)
	{
		cl_int err = clEnqueueReadBuffer(gCLQueue, mMemory, CL_TRUE, 0, mSize * sizeof(cl_float), mData, 0, NULL, NULL);
		if (err != CL_SUCCESS)
		{
			printf("ERROR: copytoCPU: %d\n", err);
		}
	}
	/*cl_int err = clblasReadMatrix(clblasRowMajor, mSize * sizeof(cl_float), mSize * sizeof(cl_float), sizeof(cl_float),
	mMemory, 0, cols(), mData, 0, cols(),
	gCLQueue, 1, NULL);
	if (err != CL_SUCCESS)
	{
	printf("ERROR: copytoGPU: %d\n", err);
	}*/
}

void Tensor::setzero()
{
	memset(mData, 0, sizeof(Float)*mSize);
}

void Tensor::setconstant(Float c)
{
	for (uint64_t i = 0; i < mSize; i++)
	{
		mData[i] = c;
	}
}

void Tensor::setidentity()
{
	setzero();
	assert(mShape.size() == 2 && "Not a matrix");
	assert(mShape[0] == mShape[1] && "Not a square matrix");
	for (uint64_t i = 0; i < mShape[0]; i++)
	{
		operator()(i, i) = 1;
	}
}

void Tensor::reshape(const TensorShape& shape)
{
	mShape = shape;
	mLD = shape[mShape.size()-1];
}

Float Tensor::sum()
{
	Float res = 0.0;
	for (uint64_t i = 0; i < mSize; i++)
	{
		res += mData[i];
	}
	return res;
}

//Tensor Tensor::subtensor(const TensorShape& begin, const TensorShape& size)
//{
//	assert(begin.size() == mShape.size() && size.size() == mShape.size());
//	unsigned int ptr = 0;
//	for (unsigned int i = 0; i <= begin.size(); i++)
//	{
//		ptr *= mShape[i];
//		ptr += begin[i];
//	}
//	return Tensor(&mData[ptr], size);
//}

Tensor Tensor::cut(uint64_t begin, uint64_t len) const
{
	//printf("%d %d %d\n", begin, len, mShape[0]);
#ifdef NN_DEBUG
	assert(begin + len <= mShape[0]);
#endif
	TensorShape shape = mShape;
	shape[0] = len;
	return Tensor(&mData[begin*(mSize/mShape[0])], shape);
}

Tensor Tensor::cut2(uint64_t begin, uint64_t len) const
{
#ifdef NN_DEBUG
	assert(begin + len <= mShape[1]);
#endif
	TensorShape shape = mShape;
	shape[1] = len;
	return Tensor(&mData[begin], shape, mLD);
}

Tensor Tensor::submatrix(uint64_t begin_row, uint64_t begin_col, uint64_t rows, uint64_t cols) const
{
#ifdef NN_DEBUG
	assert(mShape.size() == 2);
	assert(begin_row + rows <= mShape[0]);
	assert(begin_col + cols <= mShape[1]);
#endif
	TensorShape shape = make_shape(rows, cols);
	return Tensor(&mData[begin_row*mLD + begin_col], shape, mLD);
}

uint64_t Tensor::rows() const
{
#ifdef NN_DEBUG
	assert(mShape.size() >= 1);
#endif
	return mShape[0];
}

uint64_t Tensor::cols() const
{
#ifdef NN_DEBUG
	assert(mShape.size() >= 2);
#endif
	return mShape[1];
}

void Tensor::print() const
{
	for (int i = 0; i < mShape[0]; i++)
	{
		for (int j = 0; j < mShape[1]; j++)
		{
			printf("%f ", operator()(i, j));
		}
		printf("\n");
	}
}

void Tensor::print_raw() const
{
	for (int i = 0; i < mSize; i++)
	{
		printf("%f ", operator()(i));
	}
	printf("\n");
}
