cmake_minimum_required(VERSION 2.6)

# include(${CMAKE_BINARY_DIR}/CMakeTools/CMakeToolsHelpers.cmake)

project(CoreNN)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -funsafe-math-optimizations")

SET(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Bin)
# SET(CMAKE_BUILD_TYPE RELEASE)

# SET(PROJECT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/Source)

#find_package( Curses REQUIRED )
#include_directories( ${CURSES_INCLUDE_DIRS} )

include_directories("Include")

include_directories("/home/rian/Code/Libraries/OpenBLAS/include")
include_directories("/opt/AMDAPPSDK-3.0/include")
include_directories("/home/rian/Code/Libraries/clBLAS-2.12.0-Linux-x64/include")

link_directories("/home/rian/Code/Libraries/OpenBLAS/lib")
link_directories("/opt/AMDAPPSDK-3.0/lib/x86_64/sdk")
link_directories("/home/rian/Code/Libraries/clBLAS-2.12.0-Linux-x64/lib64")

file(GLOB SOURCE_FILES
    Source/*.cpp
    Include/*.h

    Source/Neurons/*.cpp
    Source/Optimizers/*.cpp
    Source/Initializers/*.cpp
    Source/ErrorFunctions/*.cpp

    Include/Neurons/*.h
    Include/Optimizers/*.h
    Include/Initializers/*.h
    Include/ErrorFunctions/*.h
)

add_library(CoreNN STATIC ${SOURCE_FILES})
target_link_libraries(CoreNN libopenblas.a)
target_link_libraries(CoreNN libOpenCL.so.1)
target_link_libraries(CoreNN libclBLAS2.so)
target_link_libraries(CoreNN pthread)
#target_link_libraries( CoreNN ${CURSES_LIBRARIES} )

add_executable(CoreNN_test ${SOURCE_FILES})
target_link_libraries(CoreNN_test libopenblas.a)
target_link_libraries(CoreNN_test libOpenCL.so.1)
target_link_libraries(CoreNN_test libclBLAS2.so)
target_link_libraries(CoreNN_test pthread)
target_link_libraries(CoreNN_test libmkldnn.so)
target_link_libraries(CoreNN_test libmklml_intel.so)
target_link_libraries(CoreNN_test libiomp5.so)
#target_link_libraries( CoreNN_test ${CURSES_LIBRARIES} )
