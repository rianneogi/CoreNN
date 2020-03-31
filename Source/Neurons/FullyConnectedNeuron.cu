#include "Neurons/FullyConnectedNeuron.h"

void FullyConnectedNeuron::forwardGPU()
{
	//Multiply with weights
	gemm_gpu(&Weights->Data, &mInput->Data, &mOutput->Data, CUBLAS_OP_N, CUBLAS_OP_N, 1, 0);
	
	//Add biases to every col
	for (unsigned int i = 0; i < mInput->Data.cols();i++)
	{
		Tensor tmp = mOutput->Data.cutGPU(i,1);
		assert(tmp.mSize == Biases->Data.mSize);
		// printf("gpu ");
		// printVal<<<1, 1>>>(tmp.mStartGPU);
		saxpy_gpu(&Biases->Data, &tmp, 1.0f, 1, 1);
		// printVal<<<1, 1>>>(Biases->Data.mStartGPU);
		// printVal<<<1, 1>>>(tmp.mStartGPU);
		// cudaDeviceSynchronize();
	}

	//CPU
	// gemm_cpu(&Weights->Data, &mInput->Data, &mOutput->Data, CblasNoTrans, CblasNoTrans, 1, 0);
	// for (unsigned int i = 0; i < mInput->Data.cols(); i++)
	// {
	// 	// printf("cpu %f\n", mOutput->Data(i, 0));
	// 	for (unsigned int j = 0; j < Biases->Data.mSize; j++)
	// 	{
	// 		mOutput->Data(i, j) += Biases->Data(j);
	// 	}
	// 	// printf("%f\n%f\n", Biases->Data(0), mOutput->Data(i, 0));
	// }
}

void FullyConnectedNeuron::backpropGPU()
{
	//Weights
	// printf("back1 %s\n", Name.c_str());
	gemm_gpu(&Weights->Data, &mOutput->Delta, &mInput->Delta, CUBLAS_OP_T, CUBLAS_OP_N, 1, 1);
	// printf("back2\n");
	gemm_gpu(&mOutput->Delta, &mInput->Data, &Weights->Delta, CUBLAS_OP_N, CUBLAS_OP_T, 1, 0);

	//Biases
	// printf("back3\n");
	gemm_gpu(&mOutput->Delta, &Ones, &Biases->Delta, CUBLAS_OP_N, CUBLAS_OP_N, 1, 0);


	// gemm_cpu(&Weights->Data, &mOutput->Delta, &mInput->Delta, CblasTrans, CblasNoTrans, 1, 1);
	// // printf("back2\n");
	// gemm_cpu(&mOutput->Delta, &mInput->Data, &Weights->Delta, CblasNoTrans, CblasTrans, 1, 0);

	// //Biases
	// // printf("back3\n");
	// gemm_cpu(&mOutput->Delta, &Ones, &Biases->Delta, CblasNoTrans, CblasNoTrans, 1, 0);
}