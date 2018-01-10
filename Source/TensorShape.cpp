#include "TensorShape.h"

typedef std::vector<uint64_t> TensorShape;

void print_shape(const TensorShape& shape)
{
	for (int i = 0; i < shape.size(); i++)
	{
		printf("%d ", shape[i]);
	}
	printf("\n");
}

TensorShape make_shape(uint64_t a)
{
	TensorShape shape;
	shape.push_back(a);
	return shape;
}

TensorShape make_shape(uint64_t a, uint64_t b)
{
	TensorShape shape;
	shape.push_back(a);
	shape.push_back(b);
	return shape;
}

TensorShape make_shape(uint64_t a, uint64_t b, uint64_t c)
{
	TensorShape shape;
	shape.push_back(a);
	shape.push_back(b);
	shape.push_back(c);
	return shape;
}

TensorShape make_shape(uint64_t a, uint64_t b, uint64_t c, uint64_t d)
{
	TensorShape shape;
	shape.push_back(a);
	shape.push_back(b);
	shape.push_back(c);
	shape.push_back(d);
	return shape;
}