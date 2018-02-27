#include "Neurons/FullyConnectedNeuron.h"

FullyConnectedNeuron::FullyConnectedNeuron() : Neuron()
{
}

// FullyConnectedNeuron::FullyConnectedNeuron(Blob* input, Blob* output) : FullyConnectedNeuron(input, output, -0.5, 0.5)
// {
// }

FullyConnectedNeuron::FullyConnectedNeuron(Blob* input, Blob* output, Initializer* initializer)
	: mInput(input), mOutput(output), mInitializer(initializer)
{
	// assert(input->Data.mShape.size() == 2);
	// assert(output->Data.mShape.size() == 2);
	// Weights = new Blob(make_shape(input->Data.cols(), output->Data.cols()));
	// Biases = new Blob(make_shape(1, output->Data.cols()));
	// for (int i = 0; i < Weights->Data.cols(); i++)
	// {
	// 	Biases->Data(i) = rand_init(init_start, init_end);
	// 	for (int j = 0; j < Weights->Data.rows(); j++)
	// 	{
			// Weights->Data(j, i) = rand_init(init_start, init_end);
	// 	}
	// }
	// Biases->copyToGPU();
	// Weights->copyToGPU();
	// InputSize = Weights->Data.rows();
	// OutputSize = Weights->Data.cols();
	// BatchSize = output->Data.rows();
	// assert(input->Data.rows() == output->Data.rows());
    //
	// //WeightsDelta = Tensor(make_shape(Weights->Data.rows(), Weights->Data.cols()));
	// //BiasesDelta = Tensor(make_shape(1, Biases->Data.mSize));
	// Ones = Tensor(make_shape(1, BatchSize));
	// Ones.setconstant(1);
	// Ones.copyToGPU();
}

FullyConnectedNeuron::~FullyConnectedNeuron()
{
	/*Weights.freememory();
	Biases.freememory();
	WeightsDelta.freememory();
	BiasesDelta.freememory();*/
	delete Weights;
	delete Biases;
	Ones.freemem();
}

bool FullyConnectedNeuron::init()
{
	assert(mInitializer!=nullptr);
	assert(mInput->Data.mShape.size() == 2);
	assert(mOutput->Data.mShape.size() == 2);
	Weights = new Blob(make_shape(mInput->Data.cols(), mOutput->Data.cols()), Name + "Weights");
	Biases = new Blob(make_shape(1, mOutput->Data.cols()), Name + "Bias");
	for (uint64_t i = 0; i < Weights->Data.cols(); i++)
	{
		Biases->Data(i) = mInitializer->get_value(i);
		// printf("b %d %f\n", i, Biases->Data(i));
		for (uint64_t j = 0; j < Weights->Data.rows(); j++)
		{
			Weights->Data(j, i) = mInitializer->get_value(j + i*Weights->Data.rows());
			// printf("wt %d %d %f\n", j, i, Weights->Data(j,i));
		}
	}
	Biases->copyToGPU();
	Weights->copyToGPU();
	InputSize = Weights->Data.rows();
	OutputSize = Weights->Data.cols();
	BatchSize = mOutput->Data.rows();
	assert(mInput->Data.rows() == mOutput->Data.rows());

	//WeightsDelta = Tensor(make_shape(Weights->Data.rows(), Weights->Data.cols()));
	//BiasesDelta = Tensor(make_shape(1, Biases->Data.mSize));
	Ones = Tensor(make_shape(1, BatchSize));
	Ones.setconstant(1);
	Ones.copyToGPU();

	return true;
}

void FullyConnectedNeuron::forward()
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

void FullyConnectedNeuron::backprop()
{
#ifdef USE_GPU
	//Weights
	gemm_gpu(&mOutput->Delta, &Weights->Data, &mInput->Delta, clblasNoTrans, clblasTrans, 1, 1);
	gemm_gpu(&mInput->Data, &mOutput->Delta, &Weights->Delta, clblasTrans, clblasNoTrans, 1, 0);

	//Biases
	gemm_gpu(&Ones, &mOutput->Delta, &Biases->Delta, clblasNoTrans, clblasNoTrans, 1, 0);
#else
	//Weights
	gemm_cpu(&mOutput->Delta, &Weights->Data, &mInput->Delta, CblasNoTrans, CblasTrans, 1, 1);
	gemm_cpu(&mInput->Data, &mOutput->Delta, &Weights->Delta, CblasTrans, CblasNoTrans, 1, 0);
	// printf("fc %f %f %f\n", mInput->Data(0), mOutput->Delta(0), Weights->Delta(0));
	//Biases
	gemm_cpu(&Ones, &mOutput->Delta, &Biases->Delta, CblasNoTrans, CblasNoTrans, 1, 0);
	// printf("fc bias %f %f\n", mOutput->Delta(0), Biases->Delta(0));
#endif
}

std::vector<Blob*> FullyConnectedNeuron::getVariables()
{
	std::vector<Blob*> v;
	v.push_back(Weights);
	v.push_back(Biases);
	return v;
}
