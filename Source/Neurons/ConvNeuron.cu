#include "Neurons/ConvNeuron.h"

void ConvNeuron::forwardGPU()
{
	const float alpha = 1, beta = 0;
	checkCUDNN(cudnnConvolutionForward(gCudnnHandle,
                                   &alpha,
                                   InputDesc,
                                   mInput->Data.mDataGPU,
                                   FilterDesc,
                                   Weights->Data.mDataGPU,
                                   ConvDesc,
                                   ForwardAlg,
                                   dForwardWorkspace,
                                   ForwardWorkspaceBytes,
                                   &beta,
                                   OutputDesc,
                                   mOutput->Data.mDataGPU));

}

void ConvNeuron::backpropGPU()
{
	
}