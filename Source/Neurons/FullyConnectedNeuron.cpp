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
	delete Weights;
	delete Biases;
	Ones.freemem();
}

bool FullyConnectedNeuron::init()
{
	// assert(mInitializer!=nullptr);
	assert(mInput->Data.mShape.size() == 2);
	assert(mOutput->Data.mShape.size() == 2);
	Weights = new Blob(make_shape(mInput->Data.rows(), mOutput->Data.rows()), Name + "Weights");
	Biases = new Blob(make_shape(1,mOutput->Data.rows()), Name + "Bias");
	// BiasesStacked = new Blob(mOutput->Data.mShape, Name+"BiasStacked");

	//Initialize
	for (uint64_t i = 0; i < Weights->Data.rows(); i++)
	{
		if(mInitializer==nullptr)
		{
			Biases->Data(i) = rand_init(-0.5, 0.5);
		}
		else
		{
			Biases->Data(i) = mInitializer->get_value(i);
		}
		
		// printf("b %d %f\n", i, Biases->Data(i));
		for (uint64_t j = 0; j < Weights->Data.cols(); j++)
		{
			if(mInitializer==nullptr)
			{
				Weights->Data(j, i) = rand_init(-0.5, 0.5);
			}
			else
			{
				Weights->Data(j, i) = mInitializer->get_value(j + i*Weights->Data.cols());
			}
			// BiasesStacked->Data(j, i) = Biases->Data(i);
			// printf("wt %d %d %f\n", j, i, Weights->Data(j,i));
		}
	}


	Biases->copyToGPU();
	// BiasesStacked->copyToGPU();
	Weights->copyToGPU();
	InputSize = Weights->Data.cols();
	OutputSize = Weights->Data.rows();
	BatchSize = mOutput->Data.cols();
	assert(mInput->Data.cols() == mOutput->Data.cols());

	Ones = Tensor(make_shape(1,BatchSize));
	Ones.setconstant(1);
	Ones.copyToGPU();

	return true;
}

void FullyConnectedNeuron::forward()
{
	// printf("fc forward %s\n", Name.c_str());
#ifdef USE_GPU
	forwardGPU();
	// forwardCPU();
#else
	forwardCPU();
#endif
}

void FullyConnectedNeuron::forwardCPU()
{
	gemm_cpu(&Weights->Data, &mInput->Data, &mOutput->Data, CblasNoTrans, CblasNoTrans, 1, 0);
	for (unsigned int i = 0; i < mInput->Data.cols(); i++)
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
	backpropGPU();
#else
	backpropCPU();
#endif
}

void FullyConnectedNeuron::backpropCPU()
{
	//Weights
	gemm_cpu(&Weights->Data, &mOutput->Delta, &mInput->Delta, CblasTrans, CblasNoTrans, 1, 1);
	gemm_cpu(&mOutput->Delta, &mInput->Data, &Weights->Delta, CblasNoTrans, CblasTrans, 1, 0);
	// printf("fc %f %f %f\n", mInput->Data(0), mOutput->Delta(0), Weights->Delta(0));
	//Biases
	gemm_cpu(&mOutput->Delta, &Ones, &Biases->Delta, CblasNoTrans, CblasNoTrans, 1, 0);
	// printf("fc bias %f %f\n", mOutput->Delta(0), Biases->Delta(0));
}

std::vector<Blob*> FullyConnectedNeuron::getVariables()
{
	std::vector<Blob*> v;
	v.push_back(Weights);
	v.push_back(Biases);
	return v;
}
