#pragma once

#include "Tensor.h"

class SubTensor : public Tensor
{
public:
    TensorShape mSubshape;
    uint64_t mSubsize;
    
    SubTensor();
	//Tensor(const Tensor& other);
	SubTensor(const TensorShape& shape); //initialize tensor allocated with given shape
	SubTensor(Float* data, const TensorShape& shape); //initialize tensor pointing to existing data
	SubTensor(Float* data, const TensorShape& shape, uint64_t ld); //initialize tensor pointing to existing data and specify leading dimension
	~SubTensor();
    
	Float& operator()(uint64_t a) const;
	Float& operator()(uint64_t a, uint64_t b) const;
	Float& operator()(uint64_t a, uint64_t b, uint64_t c) const;
	Float& operator()(uint64_t a, uint64_t b, uint64_t c, uint64_t d) const;
};
