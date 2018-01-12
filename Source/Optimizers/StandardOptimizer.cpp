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
		for (uint64_t j = 0; j < Variables[i]->Data.mSize; j++)
		{
			// printf("%f\n", Variables[i]->Delta(j));
			Variables[i]->Data(j) -= LearningRate*Variables[i]->Delta(j);
		}
	}
}
