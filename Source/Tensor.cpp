#include "Tensor.h"

#define USE_MALLOC

Tensor::Tensor() : mData(NULL), mSize(0), mMemory(NULL), mAllocSize(0), mLD(0), mStart(NULL)
{
}

//Tensor::Tensor(const Tensor& other) : mData(other.mData), mShape(other.mShape), mSize(other.mSize), mMemory(other.mMemory), mLD(other.mLD)
//{
//}

Tensor::Tensor(const TensorShape& shape) : mData(NULL), mShape(shape), mSize(1), mMemory(NULL), mAllocShape(shape), mLD(mAllocShape[mAllocShape.size() - 1]), mStart(NULL)
{
	assert(shape.size() <= 4 && "Max supported tensor shape is 4");
	for (unsigned int x : mShape)
	{
		mSize *= x;
		mOffset.push_back(0);
	}
	mAllocSize = mSize;
	//printf("Size : %d\n", mSize);
	allocateCPU();
#ifdef USE_GPU
	allocateGPU();
#endif
}

Tensor::Tensor(Float* data, const TensorShape& shape) : mData(data), mShape(shape), mSize(1), mMemory(NULL), mAllocShape(shape), mLD(mAllocShape[mAllocShape.size()-1])
{
	assert(shape.size() <= 4 && "Max supported tensor shape is 4");
	for (unsigned int x : mShape)
	{
		mSize *= x;
		mOffset.push_back(0);
	}
	mAllocSize = mSize;
	mStart = mData;
}

// Tensor::Tensor(Float* data, const TensorShape& shape, uint64_t ld) : mData(data), mShape(shape), mSize(1), mMemory(NULL), mLD(ld)
// {
// 	assert(shape.size() <= 4 && "Max supported tensor shape is 4");
// 	for (unsigned int x : mShape)
// 	{
// 		mSize *= x;
// 	}
// }

Tensor::Tensor(Float* data, const TensorShape& shape, const TensorShape& offset, const TensorShape& subshape) : mData(data), mAllocShape(shape), mSize(1), 
	mMemory(NULL), mShape(subshape), mOffset(offset), mAllocSize(1), mLD(mAllocShape[mAllocShape.size()-1])
{
	assert(subshape.size()==shape.size());
	assert(offset.size()==shape.size());
	
	uint64_t off = 0;
	for(size_t i = 0;i<shape.size();i++)
	{
		// printf("assert %d %d %d\n", mOffset[i], mShape[i], mAllocShape[i]);
		assert(mOffset[i] + mShape[i] <= mAllocShape[i]);
		mSize *= mShape[i];
		mAllocSize *= mAllocShape[i];
		off += mOffset[i];
		if(i != shape.size()-1)
			off *= mAllocShape[i+1];
		// printf("offstep: %d %d %d\n", off, mOffset[i], mAllocShape[i]);
	}
	mStart = &mData[off];
	// printf("mStart is %d, %d, %d, size %d\n", off, offset[0], offset[1], mAllocSize);
}

Tensor::~Tensor()
{
	//if(mSelfAllocated)
	//	freememory();
}

Float& Tensor::operator()(uint64_t a) const
{
#ifdef NN_DEBUG
	assert(a < mAllocSize);
#endif
	return mData[a];
	// return mData[(a/mShape[mShape.size()-1])*mLD + a%mShape[mShape.size()-1]];
}

Float& Tensor::operator()(uint64_t a, uint64_t b) const
{
#ifdef NN_DEBUG
	assert(mShape.size() >= 1);
	assert(a < mShape[0]);
	assert(b < mShape[1]);
#endif
	return mStart[(a)*mLD + b];
}

Float& Tensor::operator()(uint64_t a, uint64_t b, uint64_t c) const
{
#ifdef NN_DEBUG
	assert(mShape.size() >= 2);
	assert(a < mShape[0]);
	assert(b < mShape[1]);
	assert(c < mShape[2]);
#endif
	return mStart[(a)*mShape[1]*mShape[2] + (b)*mShape[2] + c];
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
	return mStart[(a)*mShape[1]*mShape[2]*mShape[3] + (b)*mShape[2]*mShape[3] + (c)*mShape[3] + d];
}

void Tensor::copyFromTensor(const Tensor& other)
{
	assert(mSize == other.mSize);
	mShape = other.mShape;
	mLD = other.mLD;
	mAllocSize = other.mAllocSize;
	mAllocShape = other.mAllocShape;
	mStart = other.mStart;
	mOffset = other.mOffset;
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
	mStart = mData;
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
		mStart = NULL;
#else
		delete[] mData;
		mData = NULL;
		mStart = NULL;
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
	mAllocShape = shape;
	mSize = mAllocSize;
	mLD = shape[mShape.size()-1];
	
	mStart = mData;
	for(size_t i = 0;i<mOffset.size();i++)
	{
		mOffset[i] = 0;
	}
}

void Tensor::reshape(const TensorShape& shape, const TensorShape& offset, const TensorShape& subshape)
{
	mAllocShape = shape;
	mShape = subshape;
	mOffset = offset;
	uint64_t off = 1;
	for(size_t i = 0;i<shape.size();i++)
	{
		assert(mOffset[i] + mShape[i] <= mAllocShape[i]);
		mSize *= mShape[i];
		off *= mOffset[i]+1;
	}
	mStart = &mData[off-1];
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

Tensor Tensor::subtensor(const TensorShape& begin, const TensorShape& size)
{
	assert(begin.size() == mShape.size() && size.size() == mShape.size());
	// unsigned int ptr = 0;
	// for (unsigned int i = 0; i <= begin.size(); i++)
	// {
	// 	ptr *= mShape[i];
	// 	ptr += begin[i];
	// }
	return Tensor(mData, mShape, begin, size);
}

Tensor Tensor::cut(uint64_t begin, uint64_t len) const
{
	//printf("%d %d %d\n", begin, len, mShape[0]);
#ifdef NN_DEBUG
	assert(begin + len <= mShape[0]);
#endif
	TensorShape shape = mShape;
	shape[0] = len;
	// return Tensor(&mData[begin*(mSize/mShape[0])], shape);
	TensorShape offset;
	offset.push_back(begin);
	for(size_t i = 1; i<mShape.size();i++)
	{
		offset.push_back(0);
	}
	return Tensor(mData, mShape, offset, shape);
}

Tensor Tensor::cut2(uint64_t begin, uint64_t len) const
{
#ifdef NN_DEBUG
	assert(begin + len <= mShape[1]);
#endif
	TensorShape shape = mShape;
	shape[1] = len;
	TensorShape offset = make_shape(0, begin);
	return Tensor(mData, mShape, offset, shape);
}

Tensor Tensor::submatrix(uint64_t begin_row, uint64_t begin_col, uint64_t rows, uint64_t cols) const
{
#ifdef NN_DEBUG
	assert(mShape.size() == 2);
	assert(begin_row + rows <= mShape[0]);
	assert(begin_col + cols <= mShape[1]);
#endif
	TensorShape shape = make_shape(rows, cols);
	TensorShape offset = make_shape(begin_row, begin_col);
	
	// #error fix this
	return Tensor(mData, mShape, offset, shape);
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
