cmake_minimum_required(VERSION 3.8.2)

# find_package(CUDA QUIET REQUIRED)
enable_language(CUDA)

# include(${CMAKE_BINARY_DIR}/CMakeTools/CMakeToolsHelpers.cmake)

# set(CUDAToolkit_ROOT "/opt/cuda/")

# find_program(GENERATOR nvcc)
# message(${GENERATOR})

project(CoreNN LANGUAGES CXX CUDA)
set(CUDA_SEPARABLE_COMPILATION ON)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -gencode arch=compute_75,code=sm_75)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -funsafe-math-optimizations")

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Bin)
# SET(CMAKE_BUILD_TYPE RELEASE)

# SET(PROJECT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/Source)

#find_package( Curses REQUIRED )
#include_directories( ${CURSES_INCLUDE_DIRS} )

include_directories("Include")

include_directories("/home/rian/Code/Libraries/OpenBLAS/include")
include_directories("/opt/cuda/include")
# include_directories("/opt/AMDAPPSDK-3.0/include")
# include_directories("/home/rian/Code/Libraries/clBLAS-2.12.0-Linux-x64/include")

link_directories("/home/rian/Code/Libraries/OpenBLAS/lib")
link_directories("/opt/cuda/lib64")
# link_directories("/opt/AMDAPPSDK-3.0/lib/x86_64/sdk")
# link_directories("/home/rian/Code/Libraries/clBLAS-2.12.0-Linux-x64/lib64")

file(GLOB SOURCE_FILES
    Source/*.cpp
    Source/*.cu
    Include/*.h

    Source/Neurons/*.cpp
    Source/Neurons/*.cu
    Source/Optimizers/*.cpp
    Source/Optimizers/*.cu
    Source/Initializers/*.cpp
    Source/Initializers/*.cu
    Source/ErrorFunctions/*.cpp
    Source/ErrorFunctions/*.cu

    Include/Neurons/*.h
    Include/Optimizers/*.h
    Include/Initializers/*.h
    Include/ErrorFunctions/*.h
)

# add_library(CoreNN STATIC ${SOURCE_FILES})
# target_link_libraries(CoreNN libopenblas.a)
# target_link_libraries(CoreNN libcuda.so)
# target_link_libraries(CoreNN libcublas.so)
# # target_link_libraries(CoreNN libOpenCL.so.1)
# # target_link_libraries(CoreNN libclBLAS2.so)
# target_link_libraries(CoreNN pthread)
# #target_link_libraries( CoreNN ${CURSES_LIBRARIES} )

add_executable(CoreNN_test ${SOURCE_FILES})
target_link_libraries(CoreNN_test libopenblas.a)
target_link_libraries(CoreNN_test libcuda.so)
target_link_libraries(CoreNN_test libcublas.so)
target_link_libraries(CoreNN_test libcurand.so)
target_link_libraries(CoreNN_test libcudnn.so)
# target_link_libraries(CoreNN_test libOpenCL.so.1)
# target_link_libraries(CoreNN_test libclBLAS2.so)
target_link_libraries(CoreNN_test pthread)
# target_link_libraries(CoreNN_test libmkldnn.so)
# target_link_libraries(CoreNN_test libmklml_intel.so)
# target_link_libraries(CoreNN_test libiomp5.so)
#target_link_libraries( CoreNN_test ${CURSES_LIBRARIES} )

set_property(TARGET CoreNN_test PROPERTY CUDA_STANDARD 11)
target_compile_features(CoreNN_test PUBLIC cxx_std_11)
# set_target_properties(CoreNN_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(CoreNN_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)