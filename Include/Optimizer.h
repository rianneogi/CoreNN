#pragma once

#include "ErrorFunction.h"

class Optimizer
{
public:
	std::vector<Blob*> Variables;

	Optimizer();
	virtual ~Optimizer();

	virtual void optimize() = 0;
	virtual void addVariable(Blob* blob);
};
