#include "Optimizers/SharedOptimizer.h"

SharedOptimizer::SharedOptimizer()
{
}

SharedOptimizer::SharedOptimizer(Float learning_rate, uint64_t num_weights) : LearningRate(learning_rate), SharedWeights(make_shape(num_weights))
{
	for (uint64_t i = 0; i < SharedWeights.mSize; i++)
	{
		SharedWeights(i) = (rand() % 1024) / 1024.0;
	}
}

SharedOptimizer::~SharedOptimizer()
{
	SharedWeights.freemem();
	for (uint64_t i = 0; i < SharingFactors.size(); i++)
	{
		SharingFactors[i].freemem();
	}
}

void SharedOptimizer::optimize()
{
	for (size_t i = 0; i < Variables.size(); i++)
	{
		for (uint64_t j = 0; j < Variables[i]->Delta.mSize; j++)
		{
			Variables[i]->Data(j) = 0;
			for (uint64_t k = 0; k < SharedWeights.mSize; k++)
			{
				Float tmp = SharedWeights(k);
				SharedWeights(k) -= LearningRate*Variables[i]->Delta(j)*SharingFactors[i](k, j);
				SharingFactors[i](k,j) -= LearningRate*Variables[i]->Delta(j)*tmp;

				Variables[i]->Data(j) += SharedWeights(k)*SharingFactors[i](k, j);
			}
		}
	}
}

void SharedOptimizer::addVariable(Blob* blob)
{
	Variables.push_back(blob);
	SharingFactors.push_back(Tensor(make_shape(SharedWeights.mSize, blob->Data.mSize)));
	for (uint64_t i = 0; i < SharingFactors[SharingFactors.size() - 1].mSize; i++)
	{
		SharingFactors[SharingFactors.size() - 1](i) = (rand() % 1024) / 1024.0;
	}
}
