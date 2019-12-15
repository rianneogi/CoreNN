#include "Initializers/RangeInitializer.h"

RangeInitializer::RangeInitializer() : Start(-0.5), End(0.5), Quanta(1024)
{
}

RangeInitializer::RangeInitializer(Float start, Float end) : Start(start), End(end), Quanta(1024)
{
	assert(start <= end);
}

RangeInitializer::~RangeInitializer()
{
}

Float RangeInitializer::get_value(uint64_t n)
{
	return ((rand() % Quanta) / (Quanta*1.0))*(End - Start) + Start;
}
