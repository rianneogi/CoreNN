#include "Optimizers/StandardOptimizer.h"

StandardOptimizer::StandardOptimizer() : LearningRate(1.0)
{
}

StandardOptimizer::StandardOptimizer(Float learning_rate) : LearningRate(learning_rate)
{
}

StandardOptimizer::~StandardOptimizer()
{
}

void StandardOptimizer::optimize()
{
	for (size_t i = 0; i < Variables.size(); i++)
	{
		// Float prev = Variables[i]->Data(0);
		for (uint64_t j = 0; j < Variables[i]->Data.mSize; j++)
		{
			Variables[i]->Data(j) -= LearningRate*Variables[i]->Delta(j);
		}
		// printf("opt %d %f %f %f %f\n", i, Variables[i]->Delta(0), Variables[i]->Data(0), prev, LearningRate);
	}
}
