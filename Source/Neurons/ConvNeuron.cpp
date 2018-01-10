#include "Neurons/ConvNeuron.h"

ConvNeuron::ConvNeuron() : Neuron(), LearningRate(1)
{
}

ConvNeuron::ConvNeuron(Blob* input, Blob* output, Float learning_rate)
	: mInput(input), mOutput(output), LearningRate(learning_rate)
{
	assert(input->Data.mShape.size() == 2);
	assert(output->Data.mShape.size() == 2);
	assert(input->Data.mShape[0] == output->Data.mShape[0]);
	BatchSize = input->Data.mShape[0];

	InputSize = input->Data.mShape[1];

	OutputDepth = output->Data.mShape[1];
	//OutputHeight = output->Data.mShape[2];
	//OutputWidth = output->Data.mShape[3];

	Weights = new Blob(make_shape(InputSize, OutputDepth));
	Biases = new Blob(make_shape(1, OutputDepth));
	for (int i = 0; i < Weights->Data.cols(); i++)
	{
		Biases->Data(i) = rand_init(-0.5, 0.5);
		for (int j = 0; j < Weights->Data.rows(); j++)
		{
			Weights->Data(j, i) = rand_init(-0.5, 0.5);
		}
	}

	/*WeightsDelta = Tensor(make_shape(Weights.rows(), Weights.cols()));
	BiasesDelta = Tensor(make_shape(1, Biases.mSize));*/
	Ones = Tensor(make_shape(1, BatchSize));
	Ones.setconstant(1);

	//assert(output->Data.cols() == FieldHeight*FieldWidth);
}

ConvNeuron::~ConvNeuron()
{
	/*Weights.freememory();
	Biases.freememory();
	WeightsDelta.freememory();
	BiasesDelta.freememory();*/
	delete Weights;
	delete Biases;
	Ones.freemem();
}

void ConvNeuron::forward()
{
#ifdef USE_GPU
	gemm_gpu(&mInput->Data, &Weights->Data, &mOutput->Data, clblasNoTrans, clblasNoTrans, 1, 0);
#else
	gemm_cpu(&mInput->Data, &Weights->Data, &mOutput->Data, CblasNoTrans, CblasNoTrans, 1, 0);
#endif
	for (unsigned int i = 0; i < mInput->Data.rows(); i++)
	{
		for (unsigned int j = 0; j < Biases->Data.mSize; j++)
		{
			mOutput->Data(i, j) += Biases->Data(j);
		}
	}
}

void ConvNeuron::backprop()
{
#ifdef USE_GPU
	//Weights
	gemm_gpu(&mOutput->Delta, &Weights->Data, &mInput->Delta, clblasNoTrans, clblasTrans, 1, 0);
	gemm_gpu(&mInput->Data, &mOutput->Delta, &Weights->Delta, clblasTrans, clblasNoTrans, 1, 0);

	//Biases
	gemm_gpu(&Ones, &mOutput->Delta, &Biases->Delta, clblasNoTrans, clblasNoTrans, 1, 0);
#else
	//Weights
	gemm_cpu(&mOutput->Delta, &Weights->Data, &mInput->Delta, CblasNoTrans, CblasTrans, 1, 1);
	gemm_cpu(&mInput->Data, &mOutput->Delta, &Weights->Delta, CblasTrans, CblasNoTrans, 1, 0);

	//Biases
	gemm_cpu(&Ones, &mOutput->Delta, &Biases->Delta, CblasNoTrans, CblasNoTrans, 1, 0);
#endif
}

std::vector<Blob*> ConvNeuron::getVariables()
{
	std::vector<Blob*> v;
	v.push_back(Weights);
	v.push_back(Biases);
	return v;
}
