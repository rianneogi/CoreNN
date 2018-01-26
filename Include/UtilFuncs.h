#pragma once

#include "Typedefs.h"
#include "Clock.h"

double clamp(double x);

inline double sigmoid(double x)
{
	return (1.0 / (1.0 + exp(-x)));
}

inline double tanh_NN(double x)
{
	return tanh(x);
}

inline double square(double x)
{
	return x*x;
}

inline double rand_init(Float start, Float end)
{
	return ((rand() % 1024) / 1024.0)*(end-start) + start;
}
