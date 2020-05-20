#pragma once

#include "NNInclude.h"

void cublas_vector_add();
void test_cublas_matmul();
void test_cugemm();
void test_cugemm_symm();
void test_cudnn_conv();
void test_cudnn_forward();

// boost::gil::rgb8_image_t load_image(std::string path)
// {
// 	boost::gil::rgb8_image_t img;
// 	// boost::gil::read_image( "tensorflow.png", img, boost::gil::tiff_tag() );
// 	return img;
// }