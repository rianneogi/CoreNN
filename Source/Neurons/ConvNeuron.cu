#include "Neurons/ConvNeuron.h"

void ConvNeuron::forwardGPU()
{
#ifdef NN_DEBUG
    assert(mInput->Data.mDataGPU != NULL);
    assert(Weights->Data.mDataGPU != NULL);
    assert(mOutput->Data.mDataGPU != NULL);
#endif

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
    // printf("conv forward\n");
}

void ConvNeuron::backpropGPU()
{
	
}