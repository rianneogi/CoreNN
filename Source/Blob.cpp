#include "Blob.h"

Blob::Blob()
{
	printf("WARNING: initializing blob with default contructor\n");
}

Blob::Blob(const TensorShape& shape) : Data(shape), Delta(shape)
{
}

Blob::Blob(Tensor data, Tensor delta) : Data(data), Delta(delta)
{
	assert(data.mSize == delta.mSize);
}

Blob::~Blob()
{
	Data.freemem();
	Delta.freemem();
}

void Blob::copyToGPU()
{
	Data.copyToGPU();
	Delta.copyToGPU();
}

void Blob::copyToCPU()
{
	Data.copyToCPU();
	Delta.copyToCPU();
}

void Blob::reshape(const TensorShape& shape)
{
	Data.reshape(shape);
	Delta.reshape(shape);
}

Blob* Blob::cut(uint64_t start, uint64_t len) const
{
	return (new Blob(Data.cut(start, len), Delta.cut(start, len)));
}

Blob* Blob::cut2(uint64_t start, uint64_t len) const
{
	return (new Blob(Data.cut2(start, len), Delta.cut2(start, len)));
}

Blob* Blob::submatrix(uint64_t begin_row, uint64_t begin_col, uint64_t rows, uint64_t cols) const
{
	return (new Blob(Data.submatrix(begin_row, begin_col, rows, cols), Delta.submatrix(begin_row, begin_col, rows, cols)));
}
