#include "Neurons/ConvNeuron.h"

void ConvNeuron::forwardGPU()
{
	const float alpha = 1, beta = 0;
	checkCUDNN(cudnnConvolutionForward(gCudnnHandle,
                                   &alpha,
                                   InputDesc,
                                   d_input,
                                   FilterDesc,
                                   d_kernel,
                                   ConvDesc,
                                   ForwardAlg,
                                   dForwardWorkspace,
                                   ForwardWorkspaceBytes,
                                   &beta,
                                   OutputDesc,
                                   d_output));

}

void ConvNeuron::backpropGPU()
{
	
}