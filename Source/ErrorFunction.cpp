#include "ErrorFunction.h"

ErrorFunction::ErrorFunction()
{
}

ErrorFunction::ErrorFunction(Blob* output) : mOutput(output)
{
}

ErrorFunction::ErrorFunction(Blob* output, Tensor target) : mOutput(output), mTarget(target)
{
}

ErrorFunction::~ErrorFunction()
{
}
