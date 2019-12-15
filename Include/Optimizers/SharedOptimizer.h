#pragma once

#include "../Optimizer.h"

class SharedOptimizer : public Optimizer
{
public:
	Tensor SharedWeights;
	std::vector<Tensor> SharingFactors;
	Float LearningRate;

	SharedOptimizer();
	SharedOptimizer(Float learning_rate, uint64_t num_weights);
	~SharedOptimizer();

	void optimize();
	void addVariable(Blob* blob);
};
