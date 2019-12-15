#pragma once

#include "Blob.h"

class Initializer
{
public:
	Initializer();
	virtual ~Initializer();

	virtual Float get_value(uint64_t n) = 0;
};
