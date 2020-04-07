#include "Neurons/ConvNeuron.h"

void ConvNeuron::forwardGPU()
{
#ifdef NN_DEBUG
    assert(mInput->Data.mDataGPU != NULL);
    assert(Weights->Data.mDataGPU != NULL);
    assert(mOutput->Data.mDataGPU != NULL);
#endif

    const float alpha = 1.0f, beta = 0.0f;
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
#ifdef NN_DEBUG
    assert(mInput->Data.mDataGPU != NULL);
    assert(Weights->Data.mDataGPU != NULL);
    assert(mOutput->Data.mDataGPU != NULL);
#endif
    const float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnConvolutionBackwardData(gCudnnHandle, &alpha, FilterDesc, Weights->Data.mDataGPU, OutputDesc, mOutput->Delta.mDataGPU, ConvDesc, BackwardDataAlg,
                                            dBackwardDataWorkspace, BackwardDataWorkspaceBytes, &beta, InputDesc, mInput->Delta.mDataGPU));
    checkCUDNN(cudnnConvolutionBackwardFilter(gCudnnHandle, &alpha, InputDesc, mInput->Data.mDataGPU, OutputDesc, mOutput->Delta.mDataGPU, ConvDesc, BackwardFilterAlg,
                                              dBackwardFilterWorkspace, BackwardFilterWorkspaceBytes, &beta, FilterDesc, Weights->Delta.mDataGPU));
}