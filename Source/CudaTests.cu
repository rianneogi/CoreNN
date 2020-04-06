// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>

#include "CudaTests.h"

// #include <opencv4/opencv2/opencv.hpp>

// cv::Mat load_image(const char* image_path) {
//   cv::Mat image = cv::imread(image_path);
//   image.convertTo(image, CV_32FC3);
//   cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
//   return image;
// }

// void save_image(const char* output_filename,
//                 float* buffer,
//                 int height,
//                 int width) {
//   cv::Mat output_image(height, width, CV_32FC3, buffer);
//   // Make negative values zero.
//   cv::threshold(output_image,
//                 output_image,
//                 /*threshold=*/0,
//                 /*maxval=*/0,
//                 cv::THRESH_TOZERO);
//   cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
//   output_image.convertTo(output_image, CV_8UC3);
//   cv::imwrite(output_filename, output_image);
// }

// void load_image(std::string path)
// {
// 	ILuint imageName;
// 	ilGenImages(1, &imageName);
// 	ilBindImage(imageName);
// 	ilLoadImage(path.c_str());
// }

void test_cublas_vector_add()
{
    int n = 1 << 5;
    size_t bytes = n * sizeof(float);

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b;

    h_a = (float *)malloc(bytes);
    h_b = (float *)malloc(bytes);
    h_c = (float *)malloc(bytes);
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);

    for (int i = 0; i < n;i++)
    {
        h_a[i] = i+1;
        h_b[i] = 2 * i+1;
    }

    // cublasHandle_t handle;
    // cublasCreate_v2(&handle);

    cublasSetVector(n, sizeof(float), h_a, 1, d_a, 1);
    cublasSetVector(n, sizeof(float), h_b, 1, d_b, 1);

    const float scale = 2.0f;
    cublasSaxpy(gCublasHandle, n, &scale, d_a, 1, d_b, 1);
    cublasGetVector(n, sizeof(float), d_b, 1, h_c, 1);

    for (int i = 0; i < n;i++)
    {
        printf("%f\n", h_c[i]);
    }

    // cublasDestroy(gCublasHandle);

    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);
}

// Verify our result on the CPU
// Indexing must account for the CUBLAS operating on column-major data
void verify_solution(float *a, float *b, float *c, int M, int N, int K) {
  // Tolerance for our result (floats are imperfect)
  float epsilon = 0.001f;

  // For every row...
  for (int row = 0; row < M; row++) {
    // For every column
    for (int col = 0; col < N; col++) {
      // For every element in the row-col pair...
      float temp = 0;
      for (int i = 0; i < K; i++) {
        temp += a[row + M * i] * b[col * K + i];
      }

      // Check to see if the difference falls within our tolerance
      assert(fabs(c[col * M + row] - temp) <= epsilon);
    }
  }
}

void test_cublas_matmul()
{
  // Dimensions for our matrices
  // MxK * KxN = MxN
  const int M = 1 << 9;
  const int N = 1 << 8;
  const int K = 1 << 7;

  // Pre-calculate the size (in bytes) of our matrices
  const size_t bytes_a = M * K * sizeof(float);
  const size_t bytes_b = K * N * sizeof(float);
  const size_t bytes_c = M * N * sizeof(float);

  // Vectors for the host data
  std::vector<float> h_a(M * K);
  std::vector<float> h_b(K * N);
  std::vector<float> h_c(M * N);
  
  // Allocate device memory
  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes_a);
  cudaMalloc(&d_b, bytes_b);
  cudaMalloc(&d_c, bytes_c);

  // Pseudo random number generator
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

  // Set the seed
  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

  // Fill the matrix with random numbers on the device
  curandGenerateUniform(prng, d_a, M * K);
  curandGenerateUniform(prng, d_b, K * M);

  // cuBLAS handle
  // cublasHandle_t handle;
  // cublasCreate(&handle);

  // Scalaing factors
  float alpha = 1.0f;
  float beta = 0.0f;

  // Calculate: c = (alpha*a) * b + (beta*c)
  // MxN = MxK * KxN
  // Signature: handle, operation, operation, M, N, K, alpha, A, lda, B, ldb,
  // beta, C, ldc
  cublasSgemm(gCublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_a, M, d_b, K,
              &beta, d_c, M);

  // Copy back the three matrices
  cudaMemcpy(h_a.data(), d_a, bytes_a, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_b.data(), d_b, bytes_b, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost);

  // Verify solution
  verify_solution(h_a.data(), h_b.data(), h_c.data(), M, N, K);
  std::cout << "COMPLETED SUCCESSFULLY\n";

  // Free our memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

void test_cugemm()
{
	Tensor t1(make_shape(2, 3));
	t1(0, 0) = -1;
	t1(0, 1) = 1;
	t1(0, 2) = 4;
	t1(1, 0) = -4;
	t1(1, 1) = 0;
	t1(1, 2) = -3;
	t1.print();

	Tensor t1_t(make_shape(3, 2));
	t1_t(0, 0) = -1;
	t1_t(1, 0) = 1;
	t1_t(2, 0) = 4;
	t1_t(0, 1) = -4;
	t1_t(1, 1) = 0;
	t1_t(2, 1) = -3;
	t1_t.print();

	Tensor t2(make_shape(4, 2));
	t2(0, 0) = 2;
	t2(1, 0) = 3;
	t2(2, 0) = -2;
	t2(3, 0) = 1;
	t2(0, 1) = 4;
	t2(1, 1) = 0;
	t2(2, 1) = 5;
	t2(3, 1) = 6;
	t2.print();

	Tensor t2_t(make_shape(2, 4));
	t2_t(0, 0) = 2;
	t2_t(0, 1) = 3;
	t2_t(0, 2) = -2;
	t2_t(0, 3) = 1;
	t2_t(1, 0) = 4;
	t2_t(1, 1) = 0;
	t2_t(1, 2) = 5;
	t2_t(1, 3) = 6;
	t2_t.print();

	t1.copyToGPU();
	t1_t.copyToGPU();
	t2.copyToGPU();
	t2_t.copyToGPU();

	// printf("GPU print t1\n");
	// printGPU<<<1, 1>>>(t1.mShape[0], t1.mShape[1], t1.mLD, t1.mDataGPU);

	Tensor t3(make_shape(4, 3));

	//Mat Mul
	/*clblasDgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans, t1.cols(), t2.rows(),
	t1.rows(), 1, t1.mData, t1.rows(), t2.mData, t2.rows(), 0, t3.mData, t3.rows())*/
	//cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, t1.rows(), t2.cols(),
	//	t1.cols(), 1, t1.mData, t1.cols(), t2.mData, t2.cols(), 0, t3.mData, t3.cols());
	gemm_gpu(&t1, &t2, &t3, CUBLAS_OP_N, CUBLAS_OP_N, 1, 0);
	// cudaDeviceSynchronize();
	printf("1\n");
	// printGPU<<<1, 1>>>(t3.mShape[0],t3.mShape[1],t3.mLD,t3.mDataGPU);
	// cudaDeviceSynchronize();
	t3.copyToCPU();
	t3.print();
	gemm_gpu(&t1_t, &t2_t, &t3, CUBLAS_OP_T, CUBLAS_OP_T, 1, 0);
	t3.copyToCPU();
	printf("2\n");
	t3.print();
	gemm_gpu(&t1, &t2_t, &t3, CUBLAS_OP_N, CUBLAS_OP_T, 1, 0);
	t3.copyToCPU();
	printf("3\n");
	t3.print();
	gemm_gpu(&t1_t, &t2, &t3, CUBLAS_OP_T, CUBLAS_OP_N, 1, 0);
	t3.copyToCPU();
	printf("4\n");
	t3.print();
	
	// Result should be:
	//  30  29  43  45
	// -29 -36 -19 -34

	printf("freeing mem\n");
	t1.freemem();
	t1_t.freemem();
	t2.freemem();
	t2_t.freemem();
	t3.freemem();
}

void test_cugemm_symm()
{
	Tensor t1(make_shape(2, 2));
	t1(0, 0) = -1;
	t1(0, 1) = 1;
	t1(1, 0) = -4;
	t1(1, 1) = 0;
	t1.print();

	Tensor t2(make_shape(2, 2));
	t2(0, 0) = 2;
	t2(0, 1) = 3;
	t2(1, 0) = 4;
	t2(1, 1) = 0;
	t2.print();

	t1.copyToGPU();
	// t1_t.copyToGPU();
	t2.copyToGPU();
	// t2_t.copyToGPU();

	printf("GPU print t1\n");
	printGPU<<<1, 1>>>(t1.mShape[0], t1.mShape[1], t1.mLD, t1.mDataGPU);

	Tensor t3(make_shape(2, 2));

	//Mat Mul
	/*clblasDgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans, t1.cols(), t2.rows(),
	t1.rows(), 1, t1.mData, t1.rows(), t2.mData, t2.rows(), 0, t3.mData, t3.rows())*/
	//cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, t1.rows(), t2.cols(),
	//	t1.cols(), 1, t1.mData, t1.cols(), t2.mData, t2.cols(), 0, t3.mData, t3.cols());
	gemm_gpu(&t1, &t2, &t3, CUBLAS_OP_N, CUBLAS_OP_N, 1, 0);
	// cudaDeviceSynchronize();
	printf("1\n");
	// printGPU<<<1, 1>>>(t3.mShape[0],t3.mShape[1],t3.mLD,t3.mDataGPU);
	// cudaDeviceSynchronize();
	t3.copyToCPU();
	t3.print();
	// gemm_gpu(&t1_t, &t2_t, &t3, CUBLAS_OP_T, CUBLAS_OP_T, 1, 0);
	// t3.copyToCPU();
	// printf("2\n");
	// t3.print();
	// gemm_gpu(&t1, &t2_t, &t3, CUBLAS_OP_N, CUBLAS_OP_T, 1, 0);
	// t3.copyToCPU();
	// printf("3\n");
	// t3.print();
	// gemm_gpu(&t1_t, &t2, &t3, CUBLAS_OP_T, CUBLAS_OP_N, 1, 0);
	// t3.copyToCPU();
	// printf("4\n");
	// t3.print();
	
	// Result should be:
	//  30  29  43  45
	// -29 -36 -19 -34

	printf("freeing mem\n");
	t1.freemem();
	// t1_t.freemem();
	t2.freemem();
	// t2_t.freemem();
	t3.freemem();
}

void test_cudnn_conv()
{
	ilInit();
	std::string path = "tensorflow.png";
	// load_image("tensorflow.png");
	ILuint imageName;
	ilGenImages(1, &imageName);
	ilBindImage(imageName);
	ilLoadImage(path.c_str());
	// auto view = boost::gil::view(img);

	// auto channels = view.num_channels();
	// auto dim = view.dimensions();
	// printf("dim %d %d %d %d\n", dim.x, dim.y, dim.num_dimensions, channels);

	int width = ilGetInteger(IL_IMAGE_WIDTH);
	int height = ilGetInteger(IL_IMAGE_HEIGHT);
	int channels = ilGetInteger(IL_IMAGE_CHANNELS);

	printf("dim %d %d %d\n", width, height, channels);

	Blob *input = new Blob(make_shape(1,3,width,height)); 
	Blob *output = new Blob(make_shape(1,3,width,height));
	output->Data.setconstant(255);
	output->Data.copyToGPU();
	ConvNeuron *neuron = new ConvNeuron(input, output, 3, 3, 1, 1, 1, 1);

	ILubyte* bytes = ilGetData();
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			// printf( "%s\n", "Red Value for Pixel");
			// printf( "%d ", bytes[(i*width + j)*4 + 0]);
			input->Data(0, 0, j, i) = bytes[(i * width + j) * 4 + 0];
			// printf("%s\n", "Green Value for Pixel");
			// printf( "%d\n", bytes[(i*width + j)*4 + 1]);
			input->Data(0, 1, j, i) = bytes[(i * width + j) * 4 + 1];
			// printf( "%s\n", "Blue Value for Pixel");
			// printf( "%d\n", bytes[(i*width + j)*4 + 2]);
			input->Data(0, 2, j, i) = bytes[(i * width + j) * 4 + 2];
		}
	}
	printf("copied image to input\n");
	input->Data.copyToGPU();

	// Mystery kernel
	const float kernel_template[3][3] = {
	{1,  1, 1},
	{1, -8, 1},
	{1,  1, 1}
	};

	// float h_kernel[3][3][3][3];
	for (int kernel = 0; kernel < 3; ++kernel) 
	{
		for (int channel = 0; channel < 3; ++channel) 
		{
			for (int row = 0; row < 3; ++row) 
			{
				for (int column = 0; column < 3; ++column) 
				{
					neuron->Weights->Data(kernel,channel,row,column) = kernel_template[row][column];
					neuron->Weights->Data(kernel,channel,row,column) = (rand()%1024)/1024;
				}
			}
		}
	}
	printf("set kernel\n");
	neuron->Weights->Data.copyToGPU();

	neuron->forward();
	output->Data.copyToCPU();

	ILuint imageName2;
	ilGenImages(1, &imageName2);
	ilBindImage(imageName2);
	// ilSetInteger(IL_IMAGE_WIDTH, width);
	// ilSetInteger(IL_IMAGE_HEIGHT, height);
	ILubyte *bytes2 = new ILubyte[width * height * 4];
	printf("copying output\n");
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			// printf( "%s\n", "Red Value for Pixel");
			// printf( "%d\n", bytes[(i*width + j)*4 + 0]);
			bytes2[(i * width + j) * 4 + 0] = output->Data(0,0,j,i);
			bytes2[(i * width + j) * 4 + 1] = output->Data(0,1,j,i);
			bytes2[(i * width + j) * 4 + 2] = output->Data(0,2,j,i);
			bytes2[(i * width + j) * 4 + 3] = 155;
			printf("%d ", output->Data(0, 0, j, i));
			// printf("%s\n", "Green Value for Pixel");
			// printf( "%d\n", bytes[(i*width + j)*4 + 1]);
			// input->Data(0, 1, j, i) = bytes[(i * width + j) * 4 + 1];
			// printf( "%s\n", "Blue Value for Pixel");
			// printf( "%d\n", bytes[(i*width + j)*4 + 2]);
			// input->Data(0, 2, j, i) = bytes[(i * width + j) * 4 + 2];
		}
	}
	auto error = ilGetError();
	printf("error %d\n",error);
	ilSetPixels(0, 0, 0, width, height, 4, IL_RGBA, IL_UNSIGNED_BYTE, bytes2);
	error = ilGetError();
	printf("error %d\n",error);
	// printf(iluErrorString(error));
	printf("saving\n");
	ilEnable(IL_FILE_OVERWRITE);
	ilSaveImage("output.png");
	error = ilGetError();
	printf("error %d\n",error);
	// printf("%s\n", iluErrorString(error));
	// printf("Output in output.png\n");
	// save_image("cudnn-out.png", output->Data.mData, output->Data.mShape[2], output->Data.mShape[3]);
	//extract output
}