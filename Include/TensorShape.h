#pragma once

#include "UtilFuncs.h"

typedef std::vector<uint64_t> TensorShape;

void print_shape(const TensorShape& shape);

TensorShape make_shape(uint64_t a);
TensorShape make_shape(uint64_t a, uint64_t b);
TensorShape make_shape(uint64_t a, uint64_t b, uint64_t c);
TensorShape make_shape(uint64_t a, uint64_t b, uint64_t c, uint64_t d);