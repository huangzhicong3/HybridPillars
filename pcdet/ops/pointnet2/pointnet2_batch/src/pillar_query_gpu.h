#ifndef _PILLAR_QUERY_GPU_H
#define _PILLAR_QUERY_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int pillar_query_wrapper_fast(int b, int n, int m, float radius, int nsample, 
	at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_tensor);

void pillar_query_kernel_launcher_fast(int b, int n, int m, float radius, int nsample, 
	const float *xyz, const float *new_xyz, int *idx);

#endif
