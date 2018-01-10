#pragma once

#include "../Optimizer.h"

class StandardOptimizer : public Optimizer
{
public:
	Float LearningRate;

	StandardOptimizer();
	StandardOptimizer(Float learning_rate);
	~StandardOptimizer();

	void optimize();
};