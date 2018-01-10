#pragma once

#include "../Optimizer.h"

class AdamOptimizer : public Optimizer
{
public:
	std::vector<Tensor> Momentum;
	std::vector<Tensor> Velocity;
	Float LearningRate;
	Float Beta1;
	Float Beta2;

	AdamOptimizer();
	AdamOptimizer(Float learning_rate);
	AdamOptimizer(Float learning_rate, Float beta1, Float beta2);
	~AdamOptimizer();

	void optimize();
	void addVariable(Blob* blob);
};
