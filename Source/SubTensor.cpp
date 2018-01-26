#include "SubTensor.h"

SubTensor::SubTensor()
{
    
}

SubTensor::SubTensor(Float* data, const TensorShape& shape)
{
    
}

SubTensor::SubTensor(Float* data, const TensorShape& shape, uint64_t ld)
{
    
}

SubTensor::~SubTensor()
{
    
}

Float& SubTensor::operator()(uint64_t a) const
{
#ifdef NN_DEBUG
	assert(a < mSize);
#endif
	//return mData[a];
	return mData[(a/mShape[mShape.size()-1])*mLD + a%mShape[mShape.size()-1]];
}

Float& SubTensor::operator()(uint64_t a, uint64_t b) const
{
#ifdef NN_DEBUG
	assert(mShape.size() >= 1);
	assert(a < mShape[0]);
	assert(b < mShape[1]);
#endif
	return mData[a*mLD + b];
}

Float& SubTensor::operator()(uint64_t a, uint64_t b, uint64_t c) const
{
#ifdef NN_DEBUG
	assert(mShape.size() >= 2);
	assert(a < mShape[0]);
	assert(b < mShape[1]);
	assert(c < mShape[2]);
#endif
	return mData[a*mShape[1]*mShape[2] + b*mShape[2] + c];
}

Float& SubTensor::operator()(uint64_t a, uint64_t b, uint64_t c, uint64_t d) const
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
