#include "Optimizer.h"

Optimizer::Optimizer()
{
}

Optimizer::~Optimizer()
{
}

void Optimizer::addVariable(Blob* blob)
{
	Variables.push_back(blob);
}

