#include "Tensor.h"

#define USE_MALLOC

Tensor::Tensor() : mData(NULL), mDataGPU(NULL), mSize(0), mAllocSize(0), mLD(0), mStart(NULL), mStartGPU(NULL)
{
}

//Tensor::Tensor(const Tensor& other) : mData(other.mData), mShape(other.mShape), mSize(other.mSize), mMemory(other.mMemory), mLD(other.mLD)
//{
//}

Tensor::Tensor(const TensorShape& shape) : mData(NULL), mDataGPU(NULL), mShape(shape), mSize(1), mAllocShape(shape), mLD(mAllocShape[mAllocShape.size() - 1]), mStart(NULL)
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

Tensor::Tensor(Float* data, const TensorShape& shape, bool is_gpu) : mShape(shape), mSize(1), mAllocShape(shape), mLD(mAllocShape[mAllocShape.size()-1])
{
	assert(shape.size() <= 4 && "Max supported tensor shape is 4");

	if (is_gpu)
	{
		mDataGPU = data;
		mStartGPU = data;
		mData = NULL;
		mStart = NULL;
	}
	else
	{
		mData = data;
		mStart = data;
		mDataGPU = NULL;
		mStartGPU = NULL;
	}

	for (unsigned int x : mShape)
	{
		mSize *= x;
		mOffset.push_back(0);
	}
	mAllocSize = mSize;
}

// Tensor::Tensor(Float* data, const TensorShape& shape, uint64_t ld) : mData(data), mShape(shape), mSize(1), mMemory(NULL), mLD(ld)
// {
// 	assert(shape.size() <= 4 && "Max supported tensor shape is 4");
// 	for (unsigned int x : mShape)
// 	{
// 		mSize *= x;
// 	}
// }

Tensor::Tensor(Float* data, const TensorShape& shape, const TensorShape& offset, const TensorShape& subshape, bool is_gpu) : mAllocShape(shape), mSize(1), 
	mShape(subshape), mOffset(offset), mAllocSize(1), mLD(mAllocShape[mAllocShape.size()-1])
{
	assert(subshape.size()==shape.size());
	assert(offset.size()==shape.size());
	
	if (is_gpu)
	{
		mDataGPU = data;
		mData = NULL;
		mStart = NULL;
	}
	else
	{
		mData = data;
		mDataGPU = NULL;
		mStartGPU = NULL;
	}

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

	if(is_gpu)
		mStartGPU = &mDataGPU[off];
	else
		mStart = &mData[off];
	
	// printf("mStart is %d, %d, %d, size %d\n", off, offset[0], offset[1], mAllocSize);
}

Tensor::~Tensor()
{
	//if(mSelfAllocated)
	//	freememory();
}

Float& Tensor::at(uint64_t a) const
{
	#warning todo: fix this
	uint64_t off = 0;
	TensorShape offset_rev;
	for(int i = mShape.size()-1;i>=0;i--)
	{
		offset_rev.push_back(a%mShape[i]);
		a = a/mShape[i];
	}
	
	for(size_t i = 0;i<offset_rev.size();i++)
	{
		off += offset_rev[offset_rev.size()-1-i];
		if(i != offset_rev.size()-1)
			off *= mAllocShape[i+1];
	}
	return mStart[off];
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
	return mStart[(a)*mAllocShape[1]*mAllocShape[2] + (b)*mAllocShape[2] + c];
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
	return mStart[(a)*mAllocShape[1]*mAllocShape[2]*mAllocShape[3] + (b)*mAllocShape[2]*mAllocShape[3] + (c)*mAllocShape[3] + d];
}

void Tensor::copyFromTensor(const Tensor& other)
{
	assert(mAllocSize == other.mAllocSize);
	mShape = other.mShape;
	mLD = other.mLD;
	mSize = other.mSize;
	mAllocShape = other.mAllocShape;
	mStart = other.mStart;
	mOffset = other.mOffset;
	std::memcpy(mData, other.mData, other.mAllocSize*sizeof(Float));
	cudaMemcpy(mDataGPU, other.mDataGPU, other.mAllocSize * sizeof(Float), cudaMemcpyDeviceToDevice);
}

void Tensor::copyFromSubtensor(const Tensor& other)
{
	assert(mAllocSize == other.mSize);
	mAllocShape = other.mShape;
	mShape = other.mShape;
	mSize = other.mSize;
	mLD = other.mShape[other.mShape.size()-1];
	// mSize = other.mSize;
	mStart = mData;
	for (unsigned int x : mShape)
	{
		mOffset.push_back(0);
	}
	assert(mData != NULL);
	for (uint64_t i = 0; i < mAllocSize; i++)
	{
		mData[i] = other.at(i);
	}
	// allocateGPU();
	// copyToGPU();
	// std::memcpy(mData, other.mData, other.mAllocSize*sizeof(Float));
}

void Tensor::allocateCPU()
{
	if (mData != NULL)
	{
		freeCPU();
	}
	//printf("Allocation tensor of size: %d\n", mSize);
#ifdef USE_MALLOC
	mData = (Float*)malloc(mAllocSize * sizeof(Float));
	if (mData == NULL)
	{
		printf("ERROR: Cant allocate memory for tensor, Size: %d\n", mSize);
	}
	mStart = mData;
#else
	mData = new Float[mAllocSize];
	mStart = mData;
#endif
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
	return Tensor(mData, mShape, begin, size, false);
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
	return Tensor(mData, mShape, offset, shape, false);
}

Tensor Tensor::cut2(uint64_t begin, uint64_t len) const
{
#ifdef NN_DEBUG
	assert(begin + len <= mShape[1]);
#endif
	TensorShape shape = mShape;
	shape[1] = len;
	TensorShape offset = make_shape(0, begin);
	return Tensor(mData, mShape, offset, shape, false);
}

Tensor Tensor::submatrix(uint64_t begin_row, uint64_t begin_col, uint64_t rows, uint64_t cols) const
{
#ifdef NN_DEBUG
	assert(mShape.size() == 2 && "Not a matrix");
	assert(begin_row + rows <= mShape[0]);
	assert(begin_col + cols <= mShape[1]);
#endif
	TensorShape shape = make_shape(rows, cols);
	TensorShape offset = make_shape(begin_row, begin_col);
	
	// #error fix this
	return Tensor(mData, mShape, offset, shape, false);
}

Tensor Tensor::subtensorGPU(const TensorShape& begin, const TensorShape& size)
{
	assert(begin.size() == mShape.size() && size.size() == mShape.size());
	// unsigned int ptr = 0;
	// for (unsigned int i = 0; i <= begin.size(); i++)
	// {
	// 	ptr *= mShape[i];
	// 	ptr += begin[i];
	// }
	return Tensor(mDataGPU, mShape, begin, size, true);
}

Tensor Tensor::cutGPU(uint64_t begin, uint64_t len) const
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
	return Tensor(mDataGPU, mShape, offset, shape, true);
}

Tensor Tensor::cut2GPU(uint64_t begin, uint64_t len) const
{
#ifdef NN_DEBUG
	assert(begin + len <= mShape[1]);
#endif
	TensorShape shape = mShape;
	shape[1] = len;
	TensorShape offset = make_shape(0, begin);
	return Tensor(mDataGPU, mShape, offset, shape, true);
}

Tensor Tensor::submatrixGPU(uint64_t begin_row, uint64_t begin_col, uint64_t rows, uint64_t cols) const
{
#ifdef NN_DEBUG
	assert(mShape.size() == 2 && "Not a matrix");
	assert(begin_row + rows <= mShape[0]);
	assert(begin_col + cols <= mShape[1]);
#endif
	TensorShape shape = make_shape(rows, cols);
	TensorShape offset = make_shape(begin_row, begin_col);
	
	// #error fix this
	return Tensor(mDataGPU, mShape, offset, shape, true);
}

uint64_t Tensor::rows() const
{
#ifdef NN_DEBUG
	assert(mShape.size() >= 2);
#endif
	return mShape[1];
}

uint64_t Tensor::cols() const
{
#ifdef NN_DEBUG
	assert(mShape.size() >= 1);
#endif
	return mShape[0];
}

void Tensor::print() const
{
	for (uint64_t i = 0; i < mShape[1]; i++)
	{
		for (uint64_t j = 0; j < mShape[0]; j++)
		{
			printf("%f ", operator()(j, i));
		}
		printf("\n");
	}
}

void Tensor::print_raw() const
{
	for (uint64_t i = 0; i < mSize; i++)
	{
		printf("%f ", operator()(i));
	}
	printf("\n");
}
