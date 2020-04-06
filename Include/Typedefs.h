#pragma once

#define NN_DEBUG
#define USE_GPU

#undef NDEBUG
#include "assert.h"

#include "math.h"
#include <vector>
#include <iostream>
#include <cstring>
// #include <conio>
#include "time.h"
#include <fstream>

// #include <clBLAS.h>
// #include <cuda_runtime.h>
// #include <cublas.h>
#include <cublas_v2.h>
#include <cblas.h>
#include <curand.h>
#include <thrust/reduce.h>
#include <cudnn.h>
// #include <boost/gil.hpp>
// #include <boost/gil/io/io.hpp>
// #include <boost/gil/extension/io/tiff.hpp>

#include <IL/il.h>
#include <IL/ilu.h>

// #include <stdlib.h>
// #include <stdio.h>
// #include <cstdio>
// #include <cassert>
// #include <cmath>
// #include <ctime>

// #include <mkldnn.hpp>

typedef float Float;
