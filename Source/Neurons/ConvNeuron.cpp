#include "Neurons/ConvNeuron.h"

ConvNeuron::ConvNeuron() : Neuron()
{
}

ConvNeuron::ConvNeuron(Blob* input, Blob* output, int filter_x, int filter_y, int pad_x, int pad_y, int stride_x, int stride_y, int dilation_x, int dilation_y)
	: mInput(input), mOutput(output), FieldWidth(filter_x), FieldHeight(filter_y), PaddingX(pad_x), PaddingY(pad_y), 
	  StrideX(stride_x), StrideY(stride_y), DilationX(dilation_x), DilationY(dilation_y)
{
	// #ifdef NN_DEBUG
	// assert(mInput->Data.mShape.size() == 4);
	// assert(mOutput->Data.mShape.size() == 4);
	// assert(mInput->Data.mShape[0] == mOutput->Data.mShape[0]);
	// #endif

	// BatchSize = mInput->Data.mShape[0];

	// // InputSize = input->Data.mShape[1];

	// TensorFormat = CUDNN_TENSOR_NHWC;
	// FilterFormat = CUDNN_TENSOR_NHWC;

	// if(TensorFormat==CUDNN_TENSOR_NCHW)
	// {
	// 	InputDepth = mInput->Data.mShape[1];
	// 	InputHeight = mInput->Data.mShape[2];
	// 	InputWidth = mInput->Data.mShape[3];

	// 	OutputDepth = mOutput->Data.mShape[1];
	// 	OutputHeight = mOutput->Data.mShape[2];
	// 	OutputWidth = mOutput->Data.mShape[3];
	// }
	// else if(TensorFormat==CUDNN_TENSOR_NHWC)
	// {
	// 	InputHeight = mInput->Data.mShape[1];
	// 	InputWidth = mInput->Data.mShape[2];
	// 	InputDepth = mInput->Data.mShape[3];

	// 	OutputHeight = mOutput->Data.mShape[1];
	// 	OutputWidth = mOutput->Data.mShape[2];
	// 	OutputDepth = mOutput->Data.mShape[3];
	// }
	// else
	// {
	// 	printf("ERROR: ConvNeuron unsupported tensor format\n");
	// }

	// printf("conv %d %d %d %d %d %d %d\n", BatchSize, InputDepth, InputHeight, InputWidth, OutputDepth, OutputHeight, OutputWidth);

	// if(FilterFormat==CUDNN_TENSOR_NCHW)
	// {
	// 	// printf("test1 %d %d %d %d\n",OutputDepth,InputDepth,InputHeight,InputWidth);
	// 	Weights = new Blob(make_shape(OutputDepth, InputDepth, FieldHeight, FieldWidth));
	// 	// printf("test2\n");
	// }
	// else if(FilterFormat==CUDNN_TENSOR_NHWC)
	// {
	// 	Weights = new Blob(make_shape(OutputDepth,FieldHeight,FieldWidth,InputDepth));
	// }
	// else
	// {		
	// 	printf("ERROR: ConvNeuron unsupported tensor format\n");
	// }

	// Biases = new Blob(make_shape(1, OutputDepth));
	// for (int i = 0; i < Weights->Data.cols(); i++)
	// {
	// 	Biases->Data(i) = rand_init(-0.5, 0.5);
	// 	// for (int j = 0; j < Weights->Data.rows(); j++)
	// 	// {
	// 	// 	Weights->Data(j, i) = rand_init(-0.5, 0.5);
	// 	// }
	// }

	// for (int i = 0; i < Weights->Data.mAllocSize;i++)
	// {
	// 	Weights->Data(i) = rand_init(-0.5, 0.5);
	// 	// Weights->Data(i) = 0.0f;
	// }
	// Weights->Data.copyToGPU();

	// /*WeightsDelta = Tensor(make_shape(Weights.rows(), Weights.cols()));
	// BiasesDelta = Tensor(make_shape(1, Biases.mSize));*/
	// // Ones = Tensor(make_shape(1, BatchSize));
	// // Ones.setconstant(1);
	// //assert(output->Data.cols() == FieldHeight*FieldWidth);

	// checkCUDNN(cudnnCreateTensorDescriptor(&InputDesc));
	// checkCUDNN(cudnnSetTensor4dDescriptor(InputDesc,
    //                                   /*format=*/TensorFormat,
    //                                   /*dataType=*/CUDNN_DATA_FLOAT,
    //                                   /*batch_size=*/BatchSize,
    //                                   /*channels=*/InputDepth,
    //                                   /*image_height=*/InputHeight,
    //                                   /*image_width=*/InputWidth));

	// checkCUDNN(cudnnCreateTensorDescriptor(&OutputDesc));
	// checkCUDNN(cudnnSetTensor4dDescriptor(OutputDesc,
    //                                   /*format=*/TensorFormat,
    //                                   /*dataType=*/CUDNN_DATA_FLOAT,
    //                                   /*batch_size=*/BatchSize,
    //                                   /*channels=*/OutputDepth,
    //                                   /*image_height=*/OutputHeight,
    //                                   /*image_width=*/OutputWidth));

	// checkCUDNN(cudnnCreateFilterDescriptor(&FilterDesc));
	// checkCUDNN(cudnnSetFilter4dDescriptor(FilterDesc,
    //                                   /*dataType=*/CUDNN_DATA_FLOAT,
    //                                   /*format=*/FilterFormat,
    //                                   /*out_channels=*/OutputDepth,
    //                                   /*in_channels=*/InputDepth,
    //                                   /*kernel_height=*/FieldHeight,
    //                                   /*kernel_width=*/FieldWidth));

	// checkCUDNN(cudnnCreateConvolutionDescriptor(&ConvDesc));
	// checkCUDNN(cudnnSetConvolution2dDescriptor(ConvDesc,
    //                                        /*pad_height=*/PaddingY,
    //                                        /*pad_width=*/PaddingX,
    //                                        /*vertical_stride=*/StrideY,
    //                                        /*horizontal_stride=*/StrideX,
    //                                        /*dilation_height=*/DilationY,
    //                                        /*dilation_width=*/DilationX,
    //                                        /*mode=*/CUDNN_CROSS_CORRELATION,
    //                                        /*computeType=*/CUDNN_DATA_FLOAT));

	// assert(InputDesc != NULL);
	// assert(OutputDesc != NULL);
	// assert(FilterDesc != NULL);
	// assert(ConvDesc != NULL);
	// assert(gCudnnHandle != NULL);

	// int n, c, h, w;
	// checkCUDNN(cudnnGetConvolution2dForwardOutputDim(ConvDesc, InputDesc, FilterDesc, &n, &c, &h, &w));

	// printf("conv output dims: current: %d %d %d %d, target: %d %d %d %d\n", BatchSize,OutputDepth,OutputHeight,OutputWidth,n,c,h,w);

	// assert(n == BatchSize);
	// assert(c == OutputDepth);
	// assert(h == OutputHeight);
	// assert(w == OutputWidth);

	// checkCUDNN(cudnnGetConvolutionForwardAlgorithm(gCudnnHandle,
    //                                     InputDesc,
    //                                     FilterDesc,
    //                                     ConvDesc,
    //                                     OutputDesc,
    //                                     CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
    //                                     /*memoryLimitInBytes=*/0,
    //                                     &ForwardAlg));

	// checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(gCudnnHandle, FilterDesc, OutputDesc, ConvDesc, InputDesc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &BackwardDataAlg));
	// checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(gCudnnHandle, InputDesc, OutputDesc, ConvDesc, FilterDesc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &BackwardFilterAlg));

	// // printf("Forward alg: %d\n", ForwardAlg);
	
	// checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(gCudnnHandle,
	// 												   InputDesc,
	// 												   FilterDesc,
	// 												   ConvDesc,
	// 												   OutputDesc,
	// 												   ForwardAlg,
	// 												   &ForwardWorkspaceBytes));

	// gpuErrChk(cudaMalloc(&dForwardWorkspace,ForwardWorkspaceBytes));

	// checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(gCudnnHandle, FilterDesc, OutputDesc, ConvDesc, InputDesc, BackwardDataAlg, &BackwardDataWorkspaceBytes));
	// gpuErrChk(cudaMalloc(&dBackwardDataWorkspace, BackwardDataWorkspaceBytes));

	// checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(gCudnnHandle, InputDesc, OutputDesc, ConvDesc, FilterDesc, BackwardFilterAlg, &BackwardFilterWorkspaceBytes));
	// gpuErrChk(cudaMalloc(&dBackwardFilterWorkspace, BackwardFilterWorkspaceBytes));

	// printf("Workspace sizes: %f, %f, %f MB\n", (ForwardWorkspaceBytes / 1048576.0),(BackwardDataWorkspaceBytes / 1048576.0),(BackwardFilterWorkspaceBytes / 1048576.0));
}

ConvNeuron::~ConvNeuron()
{
	/*Weights.freememory();
	Biases.freememory();
	WeightsDelta.freememory();
	BiasesDelta.freememory();*/
	delete Weights;
	delete Biases;

	checkCUDNN(cudnnDestroyTensorDescriptor(InputDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(OutputDesc));
	checkCUDNN(cudnnDestroyFilterDescriptor(FilterDesc));
	checkCUDNN(cudnnDestroyConvolutionDescriptor(ConvDesc));
	// Ones.freemem();
}

bool ConvNeuron::init()
{
	#ifdef NN_DEBUG
	assert(mInput->Data.mShape.size() == 4);
	assert(mOutput->Data.mShape.size() == 4);
	assert(mInput->Data.mShape[0] == mOutput->Data.mShape[0]);
	#endif

	BatchSize = mInput->Data.mShape[0];

	// InputSize = input->Data.mShape[1];

	TensorFormat = CUDNN_TENSOR_NHWC;
	FilterFormat = CUDNN_TENSOR_NHWC;

	if(TensorFormat==CUDNN_TENSOR_NCHW)
	{
		InputDepth = mInput->Data.mShape[1];
		InputHeight = mInput->Data.mShape[2];
		InputWidth = mInput->Data.mShape[3];

		OutputDepth = mOutput->Data.mShape[1];
		OutputHeight = mOutput->Data.mShape[2];
		OutputWidth = mOutput->Data.mShape[3];
	}
	else if(TensorFormat==CUDNN_TENSOR_NHWC)
	{
		InputHeight = mInput->Data.mShape[1];
		InputWidth = mInput->Data.mShape[2];
		InputDepth = mInput->Data.mShape[3];

		OutputHeight = mOutput->Data.mShape[1];
		OutputWidth = mOutput->Data.mShape[2];
		OutputDepth = mOutput->Data.mShape[3];
	}
	else
	{
		printf("ERROR: ConvNeuron unsupported tensor format\n");
	}

	printf("conv %d %d %d %d %d %d %d\n", BatchSize, InputDepth, InputHeight, InputWidth, OutputDepth, OutputHeight, OutputWidth);

	if(FilterFormat==CUDNN_TENSOR_NCHW)
	{
		// printf("test1 %d %d %d %d\n",OutputDepth,InputDepth,InputHeight,InputWidth);
		Weights = new Blob(make_shape(OutputDepth, InputDepth, FieldHeight, FieldWidth));
		// printf("test2\n");
	}
	else if(FilterFormat==CUDNN_TENSOR_NHWC)
	{
		Weights = new Blob(make_shape(OutputDepth,FieldHeight,FieldWidth,InputDepth));
	}
	else
	{		
		printf("ERROR: ConvNeuron unsupported tensor format\n");
	}

	Biases = new Blob(make_shape(1, OutputDepth));
	for (int i = 0; i < Weights->Data.cols(); i++)
	{
		Biases->Data(i) = rand_init(-0.5, 0.5);
		// for (int j = 0; j < Weights->Data.rows(); j++)
		// {
		// 	Weights->Data(j, i) = rand_init(-0.5, 0.5);
		// }
	}

	for (int i = 0; i < Weights->Data.mAllocSize;i++)
	{
		Weights->Data(i) = rand_init(-0.5, 0.5);
		// Weights->Data(i) = 0.0f;
	}
	Weights->Data.copyToGPU();

	/*WeightsDelta = Tensor(make_shape(Weights.rows(), Weights.cols()));
	BiasesDelta = Tensor(make_shape(1, Biases.mSize));*/
	// Ones = Tensor(make_shape(1, BatchSize));
	// Ones.setconstant(1);
	//assert(output->Data.cols() == FieldHeight*FieldWidth);

	checkCUDNN(cudnnCreateTensorDescriptor(&InputDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(InputDesc,
                                      /*format=*/TensorFormat,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/BatchSize,
                                      /*channels=*/InputDepth,
                                      /*image_height=*/InputHeight,
                                      /*image_width=*/InputWidth));

	checkCUDNN(cudnnCreateTensorDescriptor(&OutputDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(OutputDesc,
                                      /*format=*/TensorFormat,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/BatchSize,
                                      /*channels=*/OutputDepth,
                                      /*image_height=*/OutputHeight,
                                      /*image_width=*/OutputWidth));

	checkCUDNN(cudnnCreateFilterDescriptor(&FilterDesc));
	checkCUDNN(cudnnSetFilter4dDescriptor(FilterDesc,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*format=*/FilterFormat,
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

	assert(InputDesc != NULL);
	assert(OutputDesc != NULL);
	assert(FilterDesc != NULL);
	assert(ConvDesc != NULL);
	assert(gCudnnHandle != NULL);

	int n, c, h, w;
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(ConvDesc, InputDesc, FilterDesc, &n, &c, &h, &w));

	printf("conv output dims: current: %d %d %d %d, target: %d %d %d %d\n", BatchSize,OutputDepth,OutputHeight,OutputWidth,n,c,h,w);

	assert(n == BatchSize);
	assert(c == OutputDepth);
	assert(h == OutputHeight);
	assert(w == OutputWidth);

	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(gCudnnHandle,
                                        InputDesc,
                                        FilterDesc,
                                        ConvDesc,
                                        OutputDesc,
                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                        /*memoryLimitInBytes=*/0,
                                        &ForwardAlg));

	checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(gCudnnHandle, FilterDesc, OutputDesc, ConvDesc, InputDesc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &BackwardDataAlg));
	checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(gCudnnHandle, InputDesc, OutputDesc, ConvDesc, FilterDesc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &BackwardFilterAlg));

	// printf("Forward alg: %d\n", ForwardAlg);
	
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(gCudnnHandle,
													   InputDesc,
													   FilterDesc,
													   ConvDesc,
													   OutputDesc,
													   ForwardAlg,
													   &ForwardWorkspaceBytes));

	gpuErrChk(cudaMalloc(&dForwardWorkspace,ForwardWorkspaceBytes));

	checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(gCudnnHandle, FilterDesc, OutputDesc, ConvDesc, InputDesc, BackwardDataAlg, &BackwardDataWorkspaceBytes));
	gpuErrChk(cudaMalloc(&dBackwardDataWorkspace, BackwardDataWorkspaceBytes));

	checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(gCudnnHandle, InputDesc, OutputDesc, ConvDesc, FilterDesc, BackwardFilterAlg, &BackwardFilterWorkspaceBytes));
	gpuErrChk(cudaMalloc(&dBackwardFilterWorkspace, BackwardFilterWorkspaceBytes));

	printf("Workspace sizes: %f, %f, %f MB\n", (ForwardWorkspaceBytes / 1048576.0),(BackwardDataWorkspaceBytes / 1048576.0),(BackwardFilterWorkspaceBytes / 1048576.0));

	return true;
}

void ConvNeuron::forward()
{
	#ifdef USE_GPU
	forwardGPU();
	#else
	printf("WARNING: ConvNeuron cpu not implemented\n");
	#endif
// #ifdef USE_GPU
// 	gemm_gpu(&mInput->Data, &Weights->Data, &mOutput->Data, CUBLAS_OP_N, CUBLAS_OP_N, 1, 0);
// #else
// 	gemm_cpu(&mInput->Data, &Weights->Data, &mOutput->Data, CblasNoTrans, CblasNoTrans, 1, 0);
// #endif
// 	for (unsigned int i = 0; i < mInput->Data.rows(); i++)
// 	{
// 		for (unsigned int j = 0; j < Biases->Data.mSize; j++)
// 		{
// 			mOutput->Data(i, j) += Biases->Data(j);
// 		}
// 	}
}

void ConvNeuron::backprop()
{
	#ifdef USE_GPU
	backpropGPU();
	#else
	printf("WARNING: ConvNeuron cpu not implemented\n");
	#endif
	// #ifdef USE_GPU
	// 	//Weights
	// 	gemm_gpu(&mOutput->Delta, &Weights->Data, &mInput->Delta, CUBLAS_OP_N, CUBLAS_OP_T, 1, 0);
	// 	gemm_gpu(&mInput->Data, &mOutput->Delta, &Weights->Delta, CUBLAS_OP_T, CUBLAS_OP_N, 1, 0);

	// 	//Biases
	// 	gemm_gpu(&Ones, &mOutput->Delta, &Biases->Delta, CUBLAS_OP_N, CUBLAS_OP_N, 1, 0);
	// #else
	// 	//Weights
	// 	gemm_cpu(&mOutput->Delta, &Weights->Data, &mInput->Delta, CblasNoTrans, CblasTrans, 1, 1);
	// 	gemm_cpu(&mInput->Data, &mOutput->Delta, &Weights->Delta, CblasTrans, CblasNoTrans, 1, 0);

	// 	//Biases
	// 	gemm_cpu(&Ones, &mOutput->Delta, &Biases->Delta, CblasNoTrans, CblasNoTrans, 1, 0);
	// #endif
}

std::vector<Blob*> ConvNeuron::getVariables()
{
	std::vector<Blob*> v;
	v.push_back(Weights);
	v.push_back(Biases);
	return v;
}
