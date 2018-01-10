#pragma once

#include "Tensor.h"

class Blob
{
public:
	Tensor Data;
	Tensor Delta;

	Blob();
	Blob(const TensorShape& shape);
	Blob(Tensor data, Tensor delta);
	~Blob();

	void copyToGPU();
	void copyToCPU();

	void reshape(const TensorShape& shape);

	Blob* cut(uint64_t start, uint64_t len) const;
	Blob* cut2(uint64_t start, uint64_t len) const;
	Blob* submatrix(uint64_t begin_row, uint64_t begin_col, uint64_t rows, uint64_t cols) const;
};
