#include "Optimizers/AdamOptimizer.h"

AdamOptimizer::AdamOptimizer() : LearningRate(0.01), Beta1(0.9), Beta2(0.999)
{
}

AdamOptimizer::AdamOptimizer(Float learning_rate) : LearningRate(learning_rate), Beta1(0.9), Beta2(0.999)
{
}

AdamOptimizer::AdamOptimizer(Float learning_rate, Float beta1, Float beta2) : LearningRate(learning_rate), Beta1(beta1), Beta2(beta2)
{
}

AdamOptimizer::~AdamOptimizer()
{
	assert(Momentum.size() == Velocity.size());
	for (size_t i = 0; i < Momentum.size(); i++)
	{
		Momentum[i].freemem();
		Velocity[i].freemem();
	}
}

void AdamOptimizer::addVariable(Blob* blob)
{
	Variables.push_back(blob);
	Momentum.push_back(Tensor(blob->Data.mShape));
	Velocity.push_back(Tensor(blob->Data.mShape));
	Momentum[Momentum.size() - 1].setzero();
	Velocity[Velocity.size() - 1].setzero();
	Momentum[Momentum.size() - 1].copyToGPU();
	Velocity[Velocity.size() - 1].copyToGPU();
}
