#include "UtilFuncs.h"

double clamp(double x)
{
	return x > 1.0 ? 1.0 : (x < 0.0 ? 0.0 : x);
}