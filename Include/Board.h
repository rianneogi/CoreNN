#pragma once

#include "Optimizer.h"

#include "Neurons/AddNeuron.h"
#include "Neurons/MultiplyNeuron.h"
#include "Neurons/FullyConnectedNeuron.h"
#include "Neurons/ConvNeuron.h"
#include "Neurons/Im2ColNeuron.h"
#include "Neurons/TanhNeuron.h"
#include "Neurons/SigmoidNeuron.h"
#include "Neurons/LeakyReLUNeuron.h"
#include "Neurons/ReshapeNeuron.h"
#include "Neurons/DiagNeuron.h"
#include "Neurons/FileNeuron.h"
#include "Neurons/KingNeuron.h"
#include "Neurons/KnightNeuron.h"
#include "Neurons/RankNeuron.h"
#include "Neurons/StepNeuron.h"
#include "Neurons/SoftmaxNeuron.h"

#include "Optimizers/AdamOptimizer.h"
#include "Optimizers/StandardOptimizer.h"
#include "Optimizers/SharedOptimizer.h"

#include "ErrorFunctions/L1Error.h"
#include "ErrorFunctions/MeanSquaredError.h"
#include "ErrorFunctions/UnitError.h"
#include "ErrorFunctions/CrossEntropyError.h"
#include "ErrorFunctions/CategoricalCrossEntropyError.h"

#include "Initializers/RangeInitializer.h"

#include <map>
#include <string>

class Board
{
public:
	std::vector<Neuron*> mNeurons;
	std::vector<Blob*> mBlobs;
	std::vector<ErrorFunction*> mErrorFuncs;
	Optimizer* mOptimizer;
	std::vector<Tensor*> mPlaceholders;

	std::map<std::string, size_t> mNeuronNames;
	std::map<std::string, size_t> mBlobNames;

	bool mUseOptimizer;

	Board();
	~Board();

	void addNeuron(Neuron* n);
	void addNeuron(Neuron* n, std::string name);
	void addNeuronWithFixedVariables(Neuron* n);
	void addNeuronWithFixedVariables(Neuron* n, std::string name);
	Blob* newBlob(const TensorShape& shape);
	Blob* newBlob(const TensorShape& shape, std::string name);
	void addErrorFunction(ErrorFunction* err_func);
	void setOptimizer(Optimizer* optimizer);
	void addPlaceholder(Tensor* placeholder);

	bool setUp();
	void reset();

	//Tensor forward(const Tensor& input);
	Tensor forward(const std::vector<Tensor>& placeholders);
	Tensor forward(); //forwards with no placeholder
	Tensor forward(const Tensor& input1);
	Tensor forward(const Tensor& input1, const Tensor& input2);
	Tensor forward(const Tensor& input1, const Tensor& input2, const Tensor& input3);
	Tensor forward(const Tensor& input1, const Tensor& input2, const Tensor& input3, const Tensor& input4);
	//Float backprop(const Tensor& input, Tensor& output);
	//Float backprop(const Tensor& input, std::vector<Tensor>& output);
	Float backprop(const std::vector<Tensor>& placeholders);
	Float backprop(const Tensor& input1);
	Float backprop(const Tensor& input1, const Tensor& input2);
	Float backprop(const Tensor& input1, const Tensor& input2, const Tensor& input3);
	Float backprop(const Tensor& input1, const Tensor& input2, const Tensor& input3, const Tensor& input4);

	//Tensor predict(const Tensor& input);
	double train(const std::vector<Tensor>& inputs, unsigned int epochs, unsigned int batch_size);
	double train(const Tensor& inputs, const Tensor& outputs, unsigned int epochs, unsigned int batch_size);

	void save_variables(std::string filename);
	void load_variables(std::string filename);
	void copy_variables(const Board* b);

	Neuron* getNeuron(std::string name);
	size_t getNeuronID(std::string name);
	Blob* getBlob(std::string name);
	size_t getBlobID(std::string name);

	void clear_deltas();

	Blob *addFCNeuron(Blob* input, size_t layer_size);
	Blob *addConvNeuron(Blob* input, int filter_x, int filter_y, int pad_x, int pad_y, int stride_x, int stride_y, int dilation_x=1, int dilation_y=1);
	Blob *addTanhNeuron(Blob* input);
	Blob *addSigmoidNeuron(Blob* input);
	Blob *addLeakyReLUNeuron(Blob* input, float leak_factor=0.0f);
	Blob *addSoftmaxNeuron(Blob *input);
};
