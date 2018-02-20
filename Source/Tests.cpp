#include <x86intrin.h>
#include "Tests.h"

#include "NNInclude.h"

const std::string TEST_DATA_PATH = "../Data/";

//Vector binaryrep(int x, int size)
//{
//	Vector v;
//	int cnt = 0;
//	while (x != 0)
//	{
//		v.push_back(x % 2);
//		x /= 2;
//		cnt++;
//	}
//	while (cnt < size)
//	{
//		v.push_back(0);
//		cnt++;
//	}
//	return v;
//}

bool isprime(int x)
{
	if (x == 1 || x == 0) return false;
	int sq = sqrt(x);
	for (int i = 2; i <= sq; i++)
	{
		if (x%i == 0)
			return false;
	}
	return true;
}

std::vector<int> genprimes(int num)
{
	std::vector<int> primes;
	int cnt = 0;
	for (int i = 2; cnt < num; i++)
	{
		if (isprime(i))
		{
			primes.push_back(i);
			cnt++;
		}
	}
	return primes;
}

Tensor openidx_input(std::string filename)
{
	std::fstream file(filename, std::ios::in | std::ios::binary);
	if (!(file.is_open()))
	{
		printf("unable to open file: %s\n", filename.c_str());
		return Tensor();
	}

	file.seekg(4, std::ios::beg);
	uint32_t num = 0, row = 0, col = 0;
	char* data_32 = new char[4];

	file.read(data_32, 4);
	memcpy(&num, data_32, 4);
	num = __builtin_bswap32(num);

	file.read(data_32, 4);
	memcpy(&row, data_32, 4);
	row = __builtin_bswap32(row);

	file.read(data_32, 4);
	memcpy(&col, data_32, 4);
	col = __builtin_bswap32(col);

	//file >> num >> row >> col;
	printf("input size: %d %d %d\n", num, row, col);
	Tensor result(make_shape(num, row*col));
	unsigned char tmp;
	char byte;
	for (size_t i = 0; i < num; i++)
	{
		//result.push_back(Vector(row*col));
		for (size_t j = 0; j < row*col; j++)
		{
			//file >> byte;
			file.read(&byte, 1);
			memcpy(&tmp, &byte, 1);
			//byte = _byteswap_ushort(byte);
			//result[i][j] = tmp;
			result(i, j) = (tmp) / 256.0;
			//printf("%f\n", result(j,i));
		}
		if (i % 1000 == 0)
			printf("num: %d\n", i);
	}
	file.close();
	delete[] data_32;
	result.copyToGPU();
	return result;
}

Tensor openidx_output(std::string filename, size_t output_size)
{
	std::fstream file(filename, std::ios::in | std::ios::binary);
	if (!(file.is_open()))
	{
		printf("unable to open file: %s\n", filename.c_str());
		return Tensor();
	}

	file.seekg(4, std::ios::beg);
	uint32_t num;
	char* data_32 = new char[4];
	file.read(data_32, 4);
	memcpy(&num, data_32, 4);
	num = __builtin_bswap32(num);
	//file >> num;
	printf("output size: %d\n", num);
	Tensor result(make_shape(num, 10));
	for (size_t i = 0; i < num; i++)
	{
		//result.push_back(Vector(output_size));
		char byte;
		file.read(&byte, 1);
		//file >> byte;
		for (size_t j = 0; j < output_size; j++)
		{
			if (j == byte)
			{
				result(i, j) = 1.0;
			}
			else
			{
				result(i, j) = 0.0;
			}
		}
		if (i % 1000 == 0)
			printf("num: %d\n", i);
	}
	file.close();
	delete[] data_32;
	result.copyToGPU();
	return result;
}

struct TrainingData
{
	Tensor inputs;
	Tensor outputs;

	TrainingData() {}
	TrainingData(Tensor i, Tensor o) : inputs(i), outputs(o) {}
};

TrainingData load_cifar(std::string filename)
{
	Tensor output(make_shape(10000, 10));
	Tensor input(make_shape(10000, 3072));
	std::fstream file(filename, std::ios::in | std::ios::binary);

	if (!file.is_open())
	{
		printf("cant open file: %s\n", filename.c_str());
		return TrainingData();
	}

	unsigned char tmp;
	char byte;
	for (int i = 0; i < 10000; i++)
	{
		file.read(&byte, 1);
		for (int j = 0; j < 10; j++)
		{
			if (byte == j)
			{
				output(i, j) = 1.0;
			}
			else
			{
				output(i, j) = 0.0;
			}
		}

		for (int j = 0; j < 3072; j++)
		{
			file.read(&byte, 1);
			memcpy(&tmp, &byte, 1);
			input(i, j) = tmp / 256.0;
		}

		if (i % 1000 == 0)
			printf("%d\n", i);
	}

	file.close();

	TrainingData td(input, output);
	td.inputs.copyToGPU();
	td.outputs.copyToGPU();
	return td;
}

void printinput(Tensor input)
{
	for (int i = 0; i < 28; i++)
	{
		for (int j = 0; j < 28; j++)
		{
			if (input(i * 28 + j) >= 0.1)
			{
				printf("1");
			}
			else
			{
				printf("0");
			}
		}
		printf("\n");
	}
}

void printoutput(const Tensor& output)
{
	for (int i = 0; i < output.mSize; i++)
	{
		if (output(i) == 1)
		{
			printf("%d\n", i);
		}
	}
}

unsigned int getoutput(const Tensor& output)
{
	assert(output.mSize > 0);
	double max = output(0);
	unsigned int maxid = 0;
	for (size_t i = 1; i < output.mSize; i++)
	{
		if (output(0,i) > max)
		{
			max = output(i);
			maxid = i;
		}
	}
	return maxid;
}

void test_fc()
{
	//MNIST input size: 28x28 = 784
	//CIFAR input size: 3072

	Board b;
	int batch_size = 100;
	double learning_rate = 0.0005;
	int epochs = 5;

	Initializer* initializer = new RangeInitializer();

	Blob* inputBlob = b.newBlob(make_shape(batch_size, 784));
	Blob* layer1FCBlob = b.newBlob(make_shape(batch_size, 100));
	Blob* layer1SigBlob = b.newBlob(make_shape(batch_size, 100));
	Blob* layer2FCBlob = b.newBlob(make_shape(batch_size, 50));
	Blob* layer2SigBlob = b.newBlob(make_shape(batch_size, 50));
	/*Blob* layer3FCBlob = b.newBlob(make_shape(batch_size, 12));
	Blob* layer3SigBlob = b.newBlob(make_shape(batch_size, 12));*/
	Blob* outputFCBlob = b.newBlob(make_shape(batch_size, 10));
	Blob* outputSigBlob = b.newBlob(make_shape(batch_size, 10));

	b.setOptimizer(new AdamOptimizer(learning_rate));

	b.addNeuron(new FullyConnectedNeuron(inputBlob, layer1FCBlob, initializer));

	b.addNeuron(new LeakyReLUNeuron(layer1FCBlob, layer1SigBlob, 0.05));
	b.addNeuron(new FullyConnectedNeuron(layer1SigBlob, layer2FCBlob, initializer));
	b.addNeuron(new LeakyReLUNeuron(layer2FCBlob, layer2SigBlob, 0.05));

	/*b.addNeuron(new FullyConnectedNeuron(layer2SigBlob, layer3FCBlob, learning_rate));
	b.addNeuron(new LeakyReLUNeuron(layer3FCBlob, layer3SigBlob, 0.05));*/
	b.addNeuron(new FullyConnectedNeuron(layer2SigBlob, outputFCBlob, initializer));
	b.addNeuron(new LeakyReLUNeuron(outputFCBlob, outputSigBlob, 0.05));

	b.addErrorFunction(new MeanSquaredError(outputSigBlob));

	b.addPlaceholder(&inputBlob->Data);
	b.addPlaceholder(&b.mErrorFuncs[0]->mTarget);

	b.setUp();

	Tensor inputs_train = openidx_input(TEST_DATA_PATH + "train-images.idx3-ubyte");
	Tensor outputs_train = openidx_output(TEST_DATA_PATH + "train-labels.idx1-ubyte", 10);
	Tensor inputs_test = openidx_input(TEST_DATA_PATH + "t10k-images.idx3-ubyte");
	Tensor outputs_test = openidx_output(TEST_DATA_PATH + "t10k-labels.idx1-ubyte", 10);

	/*TrainingData b1 = load_cifar("Data/cifar-10-batches-bin/data_batch_1.bin");
	TrainingData b2 = load_cifar("Data/cifar-10-batches-bin/data_batch_2.bin");
	TrainingData b3 = load_cifar("Data/cifar-10-batches-bin/data_batch_3.bin");

	TrainingData b6 = load_cifar("Data/cifar-10-batches-bin/test_batch.bin");

	Matrix inputs_test = b6.inputs;
	Matrix outputs_test = b6.outputs;*/

	b.train(inputs_train, outputs_train, epochs, batch_size);
	/*for (int i = 0; i < 10; i++)
	{
	b.train(b1.inputs, b1.outputs, 1, 100);
	b.train(b2.inputs, b2.outputs, 1, 100);
	b.train(b3.inputs, b3.outputs, 1, 100);
	}*/


	int acc = 0;
	for (size_t i = 0; i < inputs_test.rows()/batch_size; i++)
	{
		Tensor o = b.forward(inputs_test.cut(i*batch_size, batch_size));
		for (size_t j = 0; j < batch_size; j++)
		{
			unsigned int result = getoutput(o.cut(j, 1));
			unsigned int target = getoutput(outputs_test.cut(i*batch_size + j, 1));
			
			// printf("output1\n");
			// o.cut(j,1).print();
			// printf("output2\n");
			// outputs_test.cut(i*batch_size + j, 1).print();
			// printf("%d %d\n", result, target);
			if (result == target)
			{
				acc++;
			}
		}
		/*else
		{
		printinput(inputs_train.col(i));
		printoutput(nn.forward(inputs_train.col(i)));
		printf("%d %d\n", getoutput(nn.forward(inputs_train.col(i))), getoutput(outputs_train.col(i)));
		}*/
	}
	printf("Accuracy: %f\n", (acc*1.0) / inputs_test.rows());

	inputs_train.freeCPU();
	inputs_test.freeCPU();
	outputs_train.freeCPU();
	outputs_test.freeCPU();

	//nn.save("net_handwriting.txt");
	// _getch();
}

void test_conv()
{
	Board b;
	int batch_size = 100;
	int epochs = 5;
	double learning_rate = 0.0005;

	Initializer* initializer = new RangeInitializer();

	Blob* inputBlob = b.newBlob(make_shape(batch_size, 28, 28, 1));
	Blob* l1convBlob = b.newBlob(make_shape(batch_size*26*26, 9));
	Blob* l1fcBlob = b.newBlob(make_shape(batch_size * 26 * 26, 10));
	Blob* l1tanhBlob = b.newBlob(make_shape(batch_size, 26,26,10)); //reshape in tanh neuron
	//Blob* l2inputBlob = b.newBlob(make_shape(batch_size, 10 * 26 * 26));
	Blob* l2convBlob = b.newBlob(make_shape(batch_size*24*24,10*9));
	Blob* l2fcBlob = b.newBlob(make_shape(batch_size*24*24, 10));
	Blob* l2tanhBlob = b.newBlob(make_shape(batch_size, 24*24*10));
	Blob* l3fcBlob = b.newBlob(make_shape(batch_size, 10));
	Blob* l3tanhBlob = b.newBlob(make_shape(batch_size, 10));

	b.setOptimizer(new AdamOptimizer(0.005));

	b.addNeuron(new Im2ColNeuron(inputBlob, l1convBlob, 3, 3));
	b.addNeuron(new FullyConnectedNeuron(l1convBlob, l1fcBlob, initializer));
	b.addNeuron(new LeakyReLUNeuron(l1fcBlob, l1tanhBlob, 0.05));
	// b.addNeuron(new ReshapeNeuron(l1tanhBlob, make_shape(batch_size, 10 * 26 * 26)));
	b.addNeuron(new Im2ColNeuron(l1tanhBlob, l2convBlob, 3, 3));
	//l1tanhBlob->reshape(make_shape(batch_size, 10 * 26 * 26));
	b.addNeuron(new FullyConnectedNeuron(l2convBlob, l2fcBlob, initializer));
	b.addNeuron(new LeakyReLUNeuron(l2fcBlob, l2tanhBlob, 0.05));
	b.addNeuron(new FullyConnectedNeuron(l2tanhBlob, l3fcBlob, initializer));
	b.addNeuron(new LeakyReLUNeuron(l3fcBlob, l3tanhBlob, 0.05));
	b.addErrorFunction(new MeanSquaredError(l3tanhBlob));
	//l1tanhBlob->reshape(make_shape(batch_size * 26 * 26, 10));

	b.addPlaceholder(&inputBlob->Data);
	b.addPlaceholder(&b.mErrorFuncs[0]->mTarget);

	Tensor inputs_train = openidx_input(TEST_DATA_PATH + "train-images.idx3-ubyte");
	Tensor outputs_train = openidx_output(TEST_DATA_PATH + "train-labels.idx1-ubyte", 10);
	Tensor inputs_test = openidx_input(TEST_DATA_PATH + "t10k-images.idx3-ubyte");
	Tensor outputs_test = openidx_output(TEST_DATA_PATH + "t10k-labels.idx1-ubyte", 10);

	printf("Setting up\n");
	b.setUp();

	b.train(inputs_train, outputs_train, epochs, batch_size);

	int acc = 0;
	for (size_t i = 0; i < inputs_test.rows() / batch_size; i++)
	{
		Tensor o = b.forward(inputs_test.cut(i*batch_size, batch_size));
		for (int j = 0; j < batch_size; j++)
		{
			unsigned int result = getoutput(o.cut(j, 1));
			unsigned int target = getoutput(outputs_test.cut(i*batch_size + j, 1));
			if (result == target)
			{
				acc++;
			}
		}
	}
	printf("Accuracy: %f\n", (acc*1.0) / inputs_test.rows());

	inputs_train.freemem();
	inputs_test.freemem();
	outputs_train.freemem();
	outputs_test.freemem();

	// _getch();
}

void test_gemm_subtensor()
{
	Tensor ten(make_shape(9,9));
	for(int i = 0;i<9;i++)
	{
		for(int j = 0;j<9;j++)
		{
			ten(i,j) = rand()%64;
		}
	}
	
	ten(5, 5) = -1;
	ten(5, 6) = 1;
	ten(5, 7) = 4;
	ten(6, 5) = -4;
	ten(6, 6) = 0;
	ten(6, 7) = -3;
	
	ten(0, 0) = 2;
	ten(0, 1) = 3;
	ten(0, 2) = -2;
	ten(0, 3) = 1;
	ten(1, 0) = 4;
	ten(1, 1) = 0;
	ten(1, 2) = 5;
	ten(1, 3) = 6;
	ten(2, 0) = 7;
	ten(2, 1) = 8;
	ten(2, 2) = 9;
	ten(2, 3) = 10;
	
	Tensor tx = ten.subtensor(make_shape(0,0),make_shape(3,4));
	Tensor ty = ten.subtensor(make_shape(5,5),make_shape(2,3));
	
	Tensor t1(make_shape(2, 3));
	t1(0, 0) = -1;
	t1(0, 1) = 1;
	t1(0, 2) = 4;
	t1(1, 0) = -4;
	t1(1, 1) = 0;
	t1(1, 2) = -3;
	
	Tensor t2(make_shape(3, 4));
	t2(0, 0) = 2;
	t2(0, 1) = 3;
	t2(0, 2) = -2;
	t2(0, 3) = 1;
	t2(1, 0) = 4;
	t2(1, 1) = 0;
	t2(1, 2) = 5;
	t2(1, 3) = 6;
	t2(2, 0) = 7;
	t2(2, 1) = 8;
	t2(2, 2) = 9;
	t2(2, 3) = 10;
	
	Tensor t3(make_shape(2, 4));
	gemm_cpu(&t1, &t2, &t3, CblasNoTrans, CblasNoTrans, 1, 0);
	t3.print();
	gemm_cpu(&ty, &tx, &t3, CblasNoTrans, CblasNoTrans, 1, 0);
	t3.print();
}

void test_gemm()
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

	Tensor t2(make_shape(3, 4));
	t2(0, 0) = 2;
	t2(0, 1) = 3;
	t2(0, 2) = -2;
	t2(0, 3) = 1;
	t2(1, 0) = 4;
	t2(1, 1) = 0;
	t2(1, 2) = 5;
	t2(1, 3) = 6;
	t2(2, 0) = 7;
	t2(2, 1) = 8;
	t2(2, 2) = 9;
	t2(2, 3) = 10;
	t2.print();

	Tensor t2_t(make_shape(4, 3));
	t2_t(0, 0) = 2;
	t2_t(1, 0) = 3;
	t2_t(2, 0) = -2;
	t2_t(3, 0) = 1;
	t2_t(0, 1) = 4;
	t2_t(1, 1) = 0;
	t2_t(2, 1) = 5;
	t2_t(3, 1) = 6;
	t2_t(0, 2) = 7;
	t2_t(1, 2) = 8;
	t2_t(2, 2) = 9;
	t2_t(3, 2) = 10;
	t2_t.print();

	Tensor t3(make_shape(2, 4));

	//Mat Mul
	/*clblasDgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans, t1.cols(), t2.rows(),
	t1.rows(), 1, t1.mData, t1.rows(), t2.mData, t2.rows(), 0, t3.mData, t3.rows())*/
	//cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, t1.rows(), t2.cols(),
	//	t1.cols(), 1, t1.mData, t1.cols(), t2.mData, t2.cols(), 0, t3.mData, t3.cols());
	gemm_cpu(&t1, &t2, &t3, CblasNoTrans, CblasNoTrans, 1, 0);
	t3.print();
	gemm_cpu(&t1_t, &t2_t, &t3, CblasTrans, CblasTrans, 1, 0);
	t3.print();
	gemm_cpu(&t1, &t2_t, &t3, CblasNoTrans, CblasTrans, 1, 0);
	t3.print();
	gemm_cpu(&t1_t, &t2, &t3, CblasTrans, CblasNoTrans, 1, 0);
	t3.print();

	t1.freeCPU();
	t1_t.freeCPU();
	t2.freeCPU();
	t2_t.freeCPU();
	t3.freeCPU();

	// _getch();
}

void test_tensor()
{
	Tensor t1(make_shape(10, 10));
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			t1(i, j) = i * 10 + j;
		}
	}

	Tensor t2(make_shape(10, 10));
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			t2(i, j) = 100 + i * 10 + j;
		}
	}

	Tensor tx = t1.submatrix(2, 2, 2, 3);
	Tensor ty = t2.submatrix(2, 2, 3, 4);

	Tensor ti = t1.cut(2, 4);
	Tensor tj = t2.cut2(2,4);

	Tensor s(make_shape(2, 4));
	s.setzero();
	Tensor s2(make_shape(2, 4));
	s2.setzero();

	t1.print();
	t2.print();
	printf("\n\n");
	tx.print();
	ty.print();
	printf("\n\n");
	ti.print();
	tj.print();

	gemm_cpu(&tx, &ty, &s, CblasNoTrans, CblasNoTrans, 1, 0);
	gemm_cpu(&ti, &tj, &s2, CblasNoTrans, CblasNoTrans, 1, 0);

	// s.print();
	// s2.print();
	// _getch();
}

void test_im2col()
{
	Board b;
	int batch_size = 10;
	double learning_rate = 0.0005;

	Blob* inputBlob = b.newBlob(make_shape(batch_size, 10, 10, 1));
	Blob* l1convBlob = b.newBlob(make_shape(batch_size * 8 * 8, 9));
	//Blob* l1fcBlob = b.newBlob(make_shape(batch_size * 8 * 8, 10));
	b.addNeuron(new Im2ColNeuron(inputBlob, l1convBlob, 3, 3));
	//b.addNeuron(new ConvNeuron(l1convBlob, l1fcBlob, learning_rate));
	b.addPlaceholder(&inputBlob->Data);

	Tensor input(make_shape(batch_size, 1, 10, 10));

	for (int i = 0; i < batch_size; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			for (int k = 0; k < 10; k++)
			{
				input(i, 0, j, k) = i * 10 * 10 + j * 10 + k;
			}
		}
	}

	b.forward(input);
	l1convBlob->Data.print();
}

//void test_gemm_gpu()
//{
//	Tensor t1(make_shape(2, 3));
//	t1(0, 0) = -1;
//	t1(0, 1) = 1;
//	t1(0, 2) = 4;
//	t1(1, 0) = -4;
//	t1(1, 1) = 0;
//	t1(1, 2) = -3;
//	t1.print();
//
//	Tensor t1_t(make_shape(3, 2));
//	t1_t(0, 0) = -1;
//	t1_t(1, 0) = 1;
//	t1_t(2, 0) = 4;
//	t1_t(0, 1) = -4;
//	t1_t(1, 1) = 0;
//	t1_t(2, 1) = -3;
//	t1_t.print();
//
//	Tensor t2(make_shape(3, 4));
//	t2(0, 0) = 2;
//	t2(0, 1) = 3;
//	t2(0, 2) = -2;
//	t2(0, 3) = 1;
//	t2(1, 0) = 4;
//	t2(1, 1) = 0;
//	t2(1, 2) = 5;
//	t2(1, 3) = 6;
//	t2(2, 0) = 7;
//	t2(2, 1) = 8;
//	t2(2, 2) = 9;
//	t2(2, 3) = 10;
//	t2.print();
//
//	Tensor t2_t(make_shape(4, 3));
//	t2_t(0, 0) = 2;
//	t2_t(1, 0) = 3;
//	t2_t(2, 0) = -2;
//	t2_t(3, 0) = 1;
//	t2_t(0, 1) = 4;
//	t2_t(1, 1) = 0;
//	t2_t(2, 1) = 5;
//	t2_t(3, 1) = 6;
//	t2_t(0, 2) = 7;
//	t2_t(1, 2) = 8;
//	t2_t(2, 2) = 9;
//	t2_t(3, 2) = 10;
//	t2_t.print();
//
//	Tensor t3(make_shape(2, 4));
//	t3.setzero();
//
//	/*t1.allocateGPU();
//	t1_t.allocateGPU();
//	t2.allocateGPU();
//	t2_t.allocateGPU();
//	t3.allocateGPU();*/
//
//	t1.copyToGPU();
//	t2.copyToGPU();
//	t1_t.copyToGPU();
//	t2_t.copyToGPU();
//	t3.copyToGPU();
//
//	//Mat Mul
//	/*clblasDgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans, t1.cols(), t2.rows(),
//	t1.rows(), 1, t1.mData, t1.rows(), t2.mData, t2.rows(), 0, t3.mData, t3.rows())*/
//	//cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, t1.rows(), t2.cols(),
//	//	t1.cols(), 1, t1.mData, t1.cols(), t2.mData, t2.cols(), 0, t3.mData, t3.cols());
//	gemm_gpu(&t1, &t2, &t3, clblasNoTrans, clblasNoTrans, 1, 0);
//	t3.copyToCPU();
//	t3.print();
//	gemm_gpu(&t1_t, &t2_t, &t3, clblasTrans, clblasTrans, 1, 0);
//	t3.copyToCPU();
//	t3.print();
//	gemm_gpu(&t1, &t2_t, &t3, clblasNoTrans, clblasTrans, 1, 0);
//	t3.copyToCPU();
//	t3.print();
//	gemm_gpu(&t1_t, &t2, &t3, clblasTrans, clblasNoTrans, 1, 0);
//	t3.copyToCPU();
//	t3.print();
//
//	t1.freemem();
//	t2.freemem();
//	t1_t.freemem();
//	t2_t.freemem();
//	t3.freemem();
//
//	_getch();
//}

void test_kernel()
{
	Tensor m1(make_shape(2, 2));
	Tensor m2(make_shape(2, 2));
	Tensor m3(make_shape(2, 2));
	m3.setzero();

	m1(0, 0) = 1;
	m1(0, 1) = 1;
	m1(1, 0) = 3;
	m1(1, 1) = 4;

	m2(0, 0) = 6;
	m2(0, 1) = 7;
	m2(1, 0) = 8;
	m2(1, 1) = 9;

	m1.copyToGPU();
	m2.copyToGPU();
	m3.copyToGPU();
	m3.print();

	clSetKernelArg(gKernelMatAdd, 0, sizeof(cl_mem), (void*)&m1.mMemory);
	clSetKernelArg(gKernelMatAdd, 1, sizeof(cl_mem), (void*)&m2.mMemory);
	clSetKernelArg(gKernelMatAdd, 2, sizeof(cl_mem), (void*)&m3.mMemory);

	clEnqueueNDRangeKernel(gCLQueue, gKernelMatAdd, 1, NULL, &(m1.mSize), NULL, 0, NULL, NULL);

	m3.copyToCPU();
	m3.print();
}

void test_diag()
{
	Board b;
	int batch_size = 3;
	int width = 8;
	int height = 8;
	int depth = 3;
	double learning_rate = 0.0005;

	b.setOptimizer(new AdamOptimizer(learning_rate));

	Blob* inputBlob = b.newBlob(make_shape(batch_size, height, width, depth));
	Blob* l1convBlob = b.newBlob(make_shape(batch_size * 2*(width+height-1), height*depth));

	Tensor t(make_shape(depth));
	t.setzero();

	b.addNeuron(new DiagNeuron(inputBlob, l1convBlob, t));

	Tensor input(make_shape(batch_size, height, width, depth));
	for (int i = 0; i < batch_size; i++)
	{
		for (int j = 0; j < height; j++)
		{
			for (int k = 0; k < width; k++)
			{
				for (int l = 0; l < depth; l++)
				{
					input(i, j, k, l) = 1000 * (i + 1) + 100 * (j + 1) + 10 * (k + 1) + (l + 1);
				}
			}
		}
	}
	b.forward(input);

	l1convBlob->Data.print();

	printf("\n %f %f %f \n", 2*input.sum(), l1convBlob->Data.sum(), 2 * input.sum() - l1convBlob->Data.sum());

	// _getch();
}
