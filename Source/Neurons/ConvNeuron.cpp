#include "Neurons/ConvNeuron.h"

ConvNeuron::ConvNeuron() : Neuron(), LearningRate(1)
{
}

ConvNeuron::ConvNeuron(Blob* input, Blob* output, int filter_x, int filter_y, int pad_x, int pad_y, int stride_x, int stride_y, int dilation_x=1, int dilation_y=1)
	: mInput(input), mOutput(output), FieldWidth(filter_x), FieldHeight(filter_y), PaddingX(pad_x), PaddingY(pad_y), StrideX(stride_x), StrideY(stride_y), DilationX(dilation_x), DilationY(dilation_y)
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

	checkCUDNN(cudnnCreateTensorDescriptor(&InputDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(InputDesc,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/BatchSize,
                                      /*channels=*/InputDepth,
                                      /*image_height=*/InputHeight,
                                      /*image_width=*/InputWidth));

	checkCUDNN(cudnnCreateTensorDescriptor(&OutputDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(OutputDesc,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/BatchSize,
                                      /*channels=*/OutputDepth,
                                      /*image_height=*/OutputHeight,
                                      /*image_width=*/OutputWidth));

	checkCUDNN(cudnnCreateFilterDescriptor(&FilterDesc));
	checkCUDNN(cudnnSetFilter4dDescriptor(FilterDesc,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*out_channels=*/OutputDepth,
                                      /*in_channels=*/InputDepth,
                                      /*kernel_height=*/FieldHeight,
                                      /*kernel_width=*/FieldWidth));

	checkCUDNN(cudnnCreateConvolutionDescriptor(&ConvDesc));
	checkCUDNN(cudnnSetConvolution2dDescriptor(ConvDesc,
                                           /*pad_height=*/PaddingY,
                                           /*pad_width=*/PaddingX,
                                           /*vertical_stride=*/StrideY,
                                           /*horizontal_stride=*/StrideX,
                                           /*dilation_height=*/DilationY,
                                           /*dilation_width=*/DilationX,
                                           /*mode=*/CUDNN_CROSS_CORRELATION,
                                           /*computeType=*/CUDNN_DATA_FLOAT));

	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(gCudnnHandle,
                                        InputDesc,
                                        FilterDesc,
                                        ConvDesc,
                                        OutputDesc,
                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                        /*memoryLimitInBytes=*/0,
                                        &ForwardAlg));

	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(gCudnnHandle,
                                                   InputDesc,
                                                   FilterDesc,
                                                   ConvDesc,
                                                   OutputDesc,
                                                   ForwardAlg,
                                                   &ForwardWorkspaceBytes));

	printf("Forward Workspace size: %d MB\n", (ForwardWorkspaceBytes / 1048576.0));
	gpuErrChk(cudaMalloc(&dForwardWorkspace,ForwardWorkspaceBytes));
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
	gemm_gpu(&mInput->Data, &Weights->Data, &mOutput->Data, CUBLAS_OP_N, CUBLAS_OP_N, 1, 0);
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
	gemm_gpu(&mOutput->Delta, &Weights->Data, &mInput->Delta, CUBLAS_OP_N, CUBLAS_OP_T, 1, 0);
	gemm_gpu(&mInput->Data, &mOutput->Delta, &Weights->Delta, CUBLAS_OP_T, CUBLAS_OP_N, 1, 0);

	//Biases
	gemm_gpu(&Ones, &mOutput->Delta, &Biases->Delta, CUBLAS_OP_N, CUBLAS_OP_N, 1, 0);
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
