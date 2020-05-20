#include "Board.h"

Board::Board() : mOptimizer(nullptr), mUseOptimizer(true)
{
}

Board::~Board()
{
	//Free memory
	for (size_t i = 0; i < mErrorFuncs.size(); i++)
	{
		delete mErrorFuncs[i];
	}
	for (size_t i = 0; i < mNeurons.size(); i++)
	{
		delete mNeurons[i];
	}
	for (size_t i = 0; i < mBlobs.size(); i++)
	{
		delete mBlobs[i];
	}
	delete mOptimizer;
}

void Board::addNeuron(Neuron* n)
{
	addNeuron(n, std::to_string(mNeurons.size()));
}

void Board::addNeuron(Neuron* n, std::string name)
{
	// assert(mOptimizer != nullptr);
	mNeuronNames[name] = mNeurons.size();
	mNeurons.push_back(n);
	n->Name = name;
	// auto variables = n->getVariables();
	// for (size_t i = 0; i < variables.size(); i++)
	// {
	// 	mOptimizer->addVariable(variables[i]);
	// }
}

// void Board::addNeuronWithFixedVariables(Neuron* n)
// {
// 	addNeuronWithFixedVariables(n, std::to_string(mNeurons.size()));
// }
//
// void Board::addNeuronWithFixedVariables(Neuron* n, std::string name)
// {
// 	mNeuronNames[name] = mNeurons.size();
// 	mNeurons.push_back(n);
// }

Blob* Board::newBlob(const TensorShape& shape)
{
	return newBlob(shape, std::to_string(mBlobs.size()));
}

Blob* Board::newBlob(const TensorShape& shape, std::string name)
{
	Blob* b = new Blob(shape, name);
	mBlobNames[name] = mBlobs.size();
	mBlobs.push_back(b);
	return b;
}

void Board::addErrorFunction(ErrorFunction* err_func)
{
	mErrorFuncs.push_back(err_func);
}

void Board::setOptimizer(Optimizer* optimizer)
{
	mOptimizer = optimizer;
}

void Board::addPlaceholder(Tensor* placeholder)
{
	mPlaceholders.push_back(placeholder);
}

bool Board::setUp()
{
	assert(mOptimizer!=nullptr);
	bool res = true;
	for(size_t i = 0;i<mNeurons.size();i++)
	{
		if(mNeurons[i]->init() == false)
		{
			printf("ERROR: Error in set up of Neuron %d", i);
			// _getch();
			res = false;
		}
	}
	reset();

	for(size_t i = 0;i<mNeurons.size();i++)
	{
		auto variables = mNeurons[i]->getVariables();
		for (size_t j = 0; j < variables.size(); j++)
		{
			mOptimizer->addVariable(variables[j]);
		}
	}
	return res;
}

void Board::reset()
{
	for(size_t i = 0;i<mNeurons.size();i++)
	{
		mNeurons[i]->reset();
	}
}

//Tensor Board::forward(const Tensor& input)
//{
//	mNeurons[0]->mInput->Data.mData = input.mData;
//	for (size_t i = 0; i < mNeurons.size(); i++)
//	{
//		mNeurons[i]->forward();
//	}
//	return mNeurons[mNeurons.size()-1]->mOutput->Data;
//}

Tensor Board::forward(const std::vector<Tensor>& placeholders)
{
	//Set placeholders
	assert(placeholders.size() <= mPlaceholders.size());
	for (size_t i = 0; i < placeholders.size(); i++)
	{
		assert(mPlaceholders[i]->mSize==placeholders[i].mSize);
		// mPlaceholders[i]->mData = placeholders[i].mData;
		// mPlaceholders[i]->mStart = placeholders[i].mStart;
		// mPlaceholders[i]->mLD = placeholders[i].mLD;
		// mPlaceholders[i]->mOffset = placeholders[i].mOffset;
		// mPlaceholders[i]->mShape = placeholders[i].mShape;
		// mPlaceholders[i]->mAllocSize = placeholders[i].mAllocSize;
		// mPlaceholders[i]->mAllocShape = placeholders[i].mAllocShape;
		// *mPlaceholders[i] = placeholders[i];
		// *mPlaceholders[i] = Tensor(placeholders[i].mAllocShape);
		mPlaceholders[i]->copyFromSubtensor(placeholders[i]);
		mPlaceholders[i]->allocateGPU();
		mPlaceholders[i]->copyToGPU();
		
		// uint64_t x = rand()%placeholders[i].mSize;
		// // printf("place %f %f %f\n",placeholders[i].at(x), mPlaceholders[i]->mData[x], mPlaceholders[i]->mStart[x]);
		// assert(placeholders[i].at(x) == mPlaceholders[i]->mData[x]);
		#warning revert this
	}
	//Forward pass
	for (size_t i = 0; i < mNeurons.size(); i++)
	{
		mNeurons[i]->forward();
	}
	return mBlobs[mBlobs.size()-1]->Data;
}

Tensor Board::forward()
{
	std::vector<Tensor> v;
	return forward(v);
}

Tensor Board::forward(const Tensor& input1)
{
	std::vector<Tensor> v;
	v.push_back(input1);
	return forward(v);
}

Tensor Board::forward(const Tensor& input1, const Tensor& input2)
{
	std::vector<Tensor> v;
	v.push_back(input1);
	v.push_back(input2);
	return forward(v);
}

Tensor Board::forward(const Tensor& input1, const Tensor& input2, const Tensor& input3)
{
	std::vector<Tensor> v;
	v.push_back(input1);
	v.push_back(input2);
	v.push_back(input3);
	return forward(v);
}

Tensor Board::forward(const Tensor& input1, const Tensor& input2, const Tensor& input3, const Tensor& input4)
{
	std::vector<Tensor> v;
	v.push_back(input1);
	v.push_back(input2);
	v.push_back(input3);
	v.push_back(input4);
	return forward(v);
}

//Float Board::backprop(const Tensor& input, Tensor& output)
//{
//	clear_deltas();
//
//	mNeurons[0]->mInput->Data.mData = input.mData;
//	mErrorFuncs[0]->mTarget = output;
//
//	//Forward Pass
//	for (size_t i = 0; i < mNeurons.size(); i++)
//	{
//		mNeurons[i]->forward();
//	}
//
//	//Calculate Error
//	Float error = mErrorFuncs[0]->calculateError();
//
//	//Backward Pass
//	for (int i = mNeurons.size() - 1; i >= 0; i--)
//	{
//		mNeurons[i]->backprop();
//	}
//
//	return error;
//}

//Float Board::backprop(const Tensor& input, std::vector<Tensor>& output)
//{
//	clear_deltas();
//
//	mNeurons[0]->mInput->Data.mData = input.mData;
//
//	for (size_t i = 0; i < mErrorFuncs.size(); i++)
//	{
//		mErrorFuncs[i]->mTarget = output[i];
//	}
//
//	//Forward Pass
//	for (size_t i = 0; i < mNeurons.size(); i++)
//	{
//		//printf("ff: %d\n", i);
//		mNeurons[i]->forward();
//	}
//
//	//Calculate Error
//	Float error = 0;
//	for (size_t i = 0; i < mErrorFuncs.size(); i++)
//	{
//		error += mErrorFuncs[i]->calculateError();
//	}
//
//	//printf("backward\n");
//	//Backward Pass
//	for (int i = mNeurons.size() - 1; i >= 0; i--)
//	{
//		//printf("bb: %d\n", i);
//		mNeurons[i]->backprop();
//	}
//
//	return error;
//}

Float Board::backprop(const std::vector<Tensor>& placeholders)
{
	clear_deltas();

	//Set placeholders
	assert(placeholders.size() <= mPlaceholders.size());
	for (size_t i = 0; i < placeholders.size(); i++)
	{
		assert(mPlaceholders[i]->mSize==placeholders[i].mSize);
		// printf("setting placeholder %d\n", i);
		// mPlaceholders[i]->mData = placeholders[i].mData;
		// mPlaceholders[i]->mStart = placeholders[i].mStart;
		// mPlaceholders[i]->mLD = placeholders[i].mLD;
		// mPlaceholders[i]->mOffset = placeholders[i].mOffset;
		// mPlaceholders[i]->mShape = placeholders[i].mShape;
		// mPlaceholders[i]->mAllocSize = placeholders[i].mAllocSize;
		// mPlaceholders[i]->mAllocShape = placeholders[i].mAllocShape;
		// *mPlaceholders[i] = placeholders[i];
		// *mPlaceholders[i] = Tensor(placeholders[i].mAllocShape);
		mPlaceholders[i]->copyFromSubtensor(placeholders[i]);
		// printf("copying placeholder to gpu %d\n", i);
		mPlaceholders[i]->allocateGPU();
		mPlaceholders[i]->copyToGPU();

		// uint64_t x = rand()%placeholders[i].mSize;
		// // printf("place %f %f %f\n",placeholders[i].at(x), mPlaceholders[i]->mData[x], mPlaceholders[i]->mStart[x]);
		// assert(placeholders[i].at(x) == mPlaceholders[i]->mData[x]);
		#warning revert this
	}

	//Forward Pass
	for (size_t i = 0; i < mNeurons.size(); i++)
	{
		// printf("forward %d : %s %f %f\n", i, mBlobs[i]->Name.c_str(), mBlobs[i]->Data(0), mBlobs[i+1]->Data(0));
		mNeurons[i]->forward();
	}

	//Calculate Error
	Float error = mErrorFuncs[0]->calculateError();

	// printf("%f del\n", mOptimizer->Variables[0]->Delta(0));

	// Backward Pass
	for (int i = mNeurons.size() - 1; i >= 0; i--)
	{
		// printf("backprop %d : %s %f %f\n", i, mBlobs[i]->Name.c_str(), mBlobs[i]->Delta(0), mBlobs[i+1]->Delta(0));
		mNeurons[i]->backprop();
	}
	
	// for(size_t i = 0;i<mOptimizer->Variables.size();i++)
	// {
	// 	printf("var %d : %s %f\n", i, mOptimizer->Variables[i]->Name.c_str(), mOptimizer->Variables[i]->Delta(0));
	// }

	return error;
}

Float Board::backprop(const Tensor& input1)
{
	std::vector<Tensor> v;
	v.push_back(input1);
	return backprop(v);
}

Float Board::backprop(const Tensor& input1, const Tensor& input2)
{
	std::vector<Tensor> v;
	v.push_back(input1);
	v.push_back(input2);
	return backprop(v);
}

Float Board::backprop(const Tensor& input1, const Tensor& input2, const Tensor& input3)
{
	std::vector<Tensor> v;
	v.push_back(input1);
	v.push_back(input2);
	v.push_back(input3);
	return backprop(v);
}

Float Board::backprop(const Tensor& input1, const Tensor& input2, const Tensor& input3, const Tensor& input4)
{
	std::vector<Tensor> v;
	v.push_back(input1);
	v.push_back(input2);
	v.push_back(input3);
	v.push_back(input4);
	return backprop(v);
}

//Tensor Board::predict(const Tensor& input)
//{
//	mNeurons[0]->mInput->Data.mData = input.mData;
//
//	//Forward Pass
//	for (size_t i = 0; i < mNeurons.size(); i++)
//	{
//		mNeurons[i]->forward();
//	}
//	return mNeurons[mNeurons.size()-1]->mOutput->Data;
//}

double Board::train(const Tensor& inputs, const Tensor& outputs, unsigned int epochs, unsigned int batch_size)
{
	assert(mErrorFuncs.size() > 0);
	//assert(mOptimizer != nullptr);
	assert(inputs.cols() == outputs.cols());
	assert(inputs.cols() % batch_size == 0);
	
	double error = 0.0;
	printf("Started training\n");
	Clock clock;
	clock.Start();
	Tensor tmp_input;
	Tensor tmp_output;

	for (int i = 0; i < epochs; i++)
	{
		error = 0.0;
		for (uint64_t j = 0; j < inputs.cols() / batch_size; j++)
		{
			tmp_input = inputs.cut(batch_size*j, batch_size);
			tmp_output = outputs.cut(batch_size*j, batch_size);
			
			// tmp_output.print();
			
			// printf("input:\n");
			// tmp_input.print();
			// printf("\noutput\n");
			// tmp_output.print();
			// int x;
			// std::cin >> x;

			std::vector<Tensor> placeholders;
			placeholders.push_back(tmp_input);
			placeholders.push_back(tmp_output);

			// tmp_input.print();
			// printf("data %f\n", tmp_input(tmp_input.mSize/2));
			// for(uint64_t i = 0;i<tmp_input.mSize;i++)
			// {
			// 	if(tmp_input(i)>2)
			// 	{
			// 		printf("%f\n", i);
			// 	}
			// }

			error += backprop(placeholders);

			if (mUseOptimizer)
				mOptimizer->optimize();

			// printf("BATCH: %d/%d\n", j+1, inputs.cols() / batch_size);

			// placeholders.clear();
		}
		clock.Stop();
		printf("Error %d: %f, epochs per sec: %f\n", i+1, error, ((i + 1)*1.0) / clock.ElapsedSeconds());
		printf("Batches per sec: %f\n", (i+1.0)*(inputs.cols()*1.0 / batch_size) / clock.ElapsedSeconds());
	}
	printf("Done training\n");

	// tmp_input.freemem();
	// tmp_output.freemem();

	return error;
}

void Board::save_variables(std::string filename)
{
	std::fstream file(filename, std::ios::out | std::ios::binary | std::ios::trunc);
	if (!file.is_open())
	{
		printf("ERROR: Unable to open file for saving: %s\n", filename.c_str());
		return;
	}
	for (size_t i = 0; i < mOptimizer->Variables.size(); i++)
	{
		for (uint64_t j = 0; j < mOptimizer->Variables[i]->Data.mSize; j++)
		{
			//file.write((const char*)&mBoard->mOptimizer->Variables[i]->Data.mData, sizeof(Float)*mBoard->mOptimizer->Variables[i]->Data.mSize);
			file.write((const char*)&mOptimizer->Variables[i]->Data(j), sizeof(Float));
			//file << mBoard->mOptimizer->Variables[i]->Data(j);
		}
		//file << "\n";
	}
	file.close();
}

void Board::load_variables(std::string filename)
{
	std::fstream file(filename, std::ios::in | std::ios::binary);
	if (!file.is_open())
	{
		printf("ERROR: Unable to open file for loading: %s\n", filename.c_str());
		return;
	}
	char* mem = new char[sizeof(Float)];
	for (size_t i = 0; i < mOptimizer->Variables.size(); i++)
	{
		for (uint64_t j = 0; j < mOptimizer->Variables[i]->Data.mSize; j++)
		{
			file.read(mem, sizeof(Float));
			std:memcpy(&mOptimizer->Variables[i]->Data(j), mem, sizeof(Float));
			//file >> mBoard->mOptimizer->Variables[i]->Data(j);
		}
		//file << "\n";
	}
	file.close();
}

void Board::copy_variables(const Board* b)
{
	assert(mOptimizer->Variables.size()==b->mOptimizer->Variables.size());
	for (size_t i = 0; i < mOptimizer->Variables.size(); i++)
	{
		assert(mOptimizer->Variables[i]->Data.mSize==b->mOptimizer->Variables[i]->Data.mSize);
		std::memcpy(mOptimizer->Variables[i]->Data.mData, b->mOptimizer->Variables[i]->Data.mData,
			sizeof(Float)*mOptimizer->Variables[i]->Data.mSize);
	}
}

Neuron* Board::getNeuron(std::string name)
{
	return mNeurons[mNeuronNames[name]];
}

size_t Board::getNeuronID(std::string name)
{
	return mNeuronNames[name];
}

Blob* Board::getBlob(std::string name)
{
	return mBlobs[mBlobNames[name]];
}

size_t Board::getBlobID(std::string name)
{
	return mBlobNames[name];
}

void Board::clear_deltas()
{
	for (size_t i = 0; i < mBlobs.size(); i++)
	{
		mBlobs[i]->Delta.setzero();
		mBlobs[i]->Delta.copyToGPU();
	}
}

Blob* Board::addFCNeuron(Blob* input, size_t layer_size)
{
	assert(input->Data.mShape.size()==2);
	size_t BatchSize = input->Data.cols();
	Blob* b = newBlob(make_shape(BatchSize, layer_size));

	addNeuron(new FullyConnectedNeuron(input,b));

	return b;
}

Blob* Board::addSigmoidNeuron(Blob* input)
{
	Blob *b = newBlob(input->Data.mShape);
	addNeuron(new SigmoidNeuron(input, b));
	return b;
}

Blob* Board::addTanhNeuron(Blob* input)
{
	Blob *b = newBlob(input->Data.mShape);
	addNeuron(new TanhNeuron(input, b));
	return b;
}

Blob* Board::addLeakyReLUNeuron(Blob* input, float leak_factor)
{
	Blob *b = newBlob(input->Data.mShape);
	addNeuron(new LeakyReLUNeuron(input, b, leak_factor));
	return b;
}

Blob* Board::addConvNeuron(Blob* input, int filter_x, int filter_y, int pad_x, int pad_y, int stride_x, int stride_y, int dilation_x, int dilation_y)
{
	Blob *b;
	return b;
}