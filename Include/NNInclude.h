#pragma once

#include "Board.h"

#include "Neurons\AddNeuron.h"
#include "Neurons\MultiplyNeuron.h"
#include "Neurons\FullyConnectedNeuron.h"
#include "Neurons\ConvNeuron.h"
#include "Neurons\Im2ColNeuron.h"
#include "Neurons\TanhNeuron.h"
#include "Neurons\SigmoidNeuron.h"
#include "Neurons\LeakyReLUNeuron.h"
#include "Neurons\ReshapeNeuron.h"
#include "Neurons\DiagNeuron.h"
#include "Neurons\FileNeuron.h"
#include "Neurons\KingNeuron.h"
#include "Neurons\KnightNeuron.h"
#include "Neurons\RankNeuron.h"
#include "Neurons\StepNeuron.h"

#include "Optimizers\AdamOptimizer.h"
#include "Optimizers\StandardOptimizer.h"
#include "Optimizers\SharedOptimizer.h"

#include "ErrorFunctions/L1Error.h"
#include "ErrorFunctions/MeanSquaredError.h"
#include "ErrorFunctions/UnitError.h"

#include "Initializers/RangeInitializer.h"
