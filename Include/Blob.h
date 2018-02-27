#pragma once

#include "Tensor.h"

class Blob
{
public:
	std::string Name;
	
	Tensor Data;
	Tensor Delta;

	Blob();
	Blob(const TensorShape& shape);
	Blob(const TensorShape& shape, std::string name);
	Blob(Tensor data, Tensor delta);
	~Blob();

	void copyToGPU();
	void copyToCPU();

	void reshape(const TensorShape& shape);
	void reshape(const TensorShape& shape, const TensorShape& offset, const TensorShape& subshape);

	Blob* subtensor(const TensorShape& begin, const TensorShape& size);
	Blob* cut(uint64_t start, uint64_t len) const;
	Blob* cut2(uint64_t start, uint64_t len) const;
	Blob* submatrix(uint64_t begin_row, uint64_t begin_col, uint64_t rows, uint64_t cols) const;
};
