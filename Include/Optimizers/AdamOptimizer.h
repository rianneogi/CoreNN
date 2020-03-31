#pragma once

#include "../Optimizer.h"

extern Float ADAM_EPSILON;

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
	void optimizeCPU();
	void optimizeGPU();
	void addVariable(Blob *blob);
};
