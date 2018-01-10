cmake_minimum_required(VERSION 2.6)

project(CoreNN)

SET(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Bin)
# SET(CMAKE_BUILD_TYPE RELEASE)

# SET(PROJECT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/Source)

include_directories("C:\\Documents\\Code\\Projects\\CoreNN\\Include")

include_directories("C:\\Documents\\Code\\Libraries\\clBLAS-2.13.0-Windows-x64\\include")
include_directories("C:\\Program Files (x86)\\AMD APP SDK\\3.0\\include")
include_directories("C:\\Documents\\Code\\Libraries\\OpenBLAS-v0.2.19-Win64-int32\\include")

link_directories("C:\\Program Files %28x86%29\\AMD APP SDK\\3.0\\lib\\x86_64")
link_directories("C:\\Documents\\Code\\Libraries\\OpenBLAS-v0.2.19-Win64-int32\\lib")
link_directories("C:\\Documents\\Code\\Libraries\\clBLAS-2.13.0-Windows-x64\\lib64\\import")

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
target_link_libraries(CoreNN OpenCL.lib)
target_link_libraries(CoreNN libopenblas.dll.lib)
target_link_libraries(CoreNN clBLAS.lib)
