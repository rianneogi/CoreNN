#pragma once

#include "../Initializer.h"

class RangeInitializer : public Initializer
{
public:
	Float Start;
	Float End;
	uint64_t Quanta;

	RangeInitializer();
	RangeInitializer(Float start, Float end);
	~RangeInitializer();

	Float get_value(uint64_t n);
};
