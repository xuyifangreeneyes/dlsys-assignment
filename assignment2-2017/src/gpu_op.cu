#include "./c_runtime_api.h"
#include <iostream>
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

enum ElemOp {
  Add = 1,
  AddByConst,
  Mul,
  MulByConst,
  Relu,
  ReluGradient
};

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  // Compute per-row loss.
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}

__global__ void softmax_kernel(int nrow, int ncol, 
                               const float *input, float *output) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int stride_x = blockDim.x * gridDim.x;
  for (int i = x; i < nrow; i += stride_x) {
    const float *input_row = input + i * ncol;
    float *output_row = output + i * ncol;
    float maxval = *input;
    for (int j = 1; j < ncol; ++j) {
      maxval = max(maxval, input_row[j]);
    }
    for (int j = 0; j < ncol; ++j) {
      output_row[j] = exp(input_row[j] - maxval);
    }
    float sum = 0;
    for (int j = 0; j < ncol; ++j) {
      sum += output_row[j];
    }
    for (int j = 0; j < ncol; ++j) {
      output_row[j] /= sum;
    }
  }
}

__global__ void matrix_elem_add_kernel(int nrow, int ncol, 
                                       const float *matA,
                                       const float *matB,
                                       float *output) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int stride_x = blockDim.x * gridDim.x;
  int stride_y = blockDim.y * gridDim.y;
  for (int i = x; i < nrow; i += stride_x) {
    for (int j = y; j < ncol; j += stride_y) {
      output[i * ncol + j] = matA[i * ncol + j] + matB[i * ncol + j];
    }
  }
}

__global__ void matrix_elem_add_by_const_kernel(int nrow, int ncol, 
                                                const float *input,
                                                float val,
                                                float *output) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int stride_x = blockDim.x * gridDim.x;
  int stride_y = blockDim.y * gridDim.y;
  for (int i = x; i < nrow; i += stride_x) {
    for (int j = y; j < ncol; j += stride_y) {
      output[i * ncol + j] = input[i * ncol + j] + val;
    }
  }
}

__global__ void matrix_elem_mul_kernel(int nrow, int ncol, 
                                       const float *matA,
                                       const float *matB,
                                       float *output) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int stride_x = blockDim.x * gridDim.x;
  int stride_y = blockDim.y * gridDim.y;
  for (int i = x; i < nrow; i += stride_x) {
    for (int j = y; j < ncol; j += stride_y) {
      output[i * ncol + j] = matA[i * ncol + j] * matB[i * ncol + j];
    }
  }
}

__global__ void matrix_elem_mul_by_const_kernel(int nrow, int ncol, 
                                                const float *input,
                                                float val,
                                                float *output) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int stride_x = blockDim.x * gridDim.x;
  int stride_y = blockDim.y * gridDim.y;
  for (int i = x; i < nrow; i += stride_x) {
    for (int j = y; j < ncol; j += stride_y) {
      output[i * ncol + j] = input[i * ncol + j] * val;
    }
  }
}

__global__ void relu_kernel(int num, const float *input, 
                            float *output) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int stride_x = blockDim.x * gridDim.x;
  for (int i = x; i < num; i += stride_x) {
    output[i] = (input[i] > 0) ? input[i] : 0;
  }
}

__global__ void relu_gradient_kernel(int num,
                                     const float *input,
                                     const float *grad,
                                     float *output) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int stride_x = blockDim.x * gridDim.x;
  for (int i = x; i < num; i += stride_x) {
    output[i] = (input[i] > 0) ? grad[i] : 0;
  }
} 

__global__ void array_set_kernel(int num, float *data, float val) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int stride_x = blockDim.x * gridDim.x;
  for (int i = x; i < num; i += stride_x) {
    data[i] = val;
  }  
}

__global__ void broadcast_to_kernel(int in_num, int ex_num, 
                                    const float *input, 
                                    float *output) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int stride_x = blockDim.x * gridDim.x;
  for (int i = x; i < in_num; i += stride_x) {
    for (int j = 0; j < ex_num; ++j) {
      output[i + j * in_num] = input[i];
    }
  }
}

__global__ void reduce_sum_axis_zero_kernel(int reduce_num, 
                                            int remain_num,
                                            const float *input, 
                                            float *output) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int stride_x = blockDim.x * gridDim.x;
  for (int i = x; i < remain_num; i += stride_x) {
    output[i] = 0;
    for (int j = 0; j < reduce_num; ++j) {
        output[i] += input[i + j * remain_num];
    }
  }
}

void checkMatrixShape(const DLArrayHandle mat1, const DLArrayHandle mat2) {
  assert(mat1->ndim == 2 & mat2->ndim == 2);
  assert(mat1->shape[0] == mat2->shape[0] && mat1->shape[1] == mat2->shape[1]); 
}

void checkMatrixShape(const DLArrayHandle mat1,
                const DLArrayHandle mat2,
                const DLArrayHandle mat3) {
  assert(mat1->ndim == 2 & mat2->ndim == 2 & mat3->ndim == 2);
  assert(mat1->shape[0] == mat2->shape[0] && mat2->shape[0] == mat3->shape[0] && 
         mat1->shape[1] == mat2->shape[1] && mat2->shape[1] == mat3->shape[1]);
}

int DLGpuArraySet(DLArrayHandle arr, float value) { 
  int ndim = arr->ndim;
  int num = 1;
  for (int i = 0; i < ndim; ++i) {
    num *= arr->shape[i];
  }
  float *arr_data = (float *)arr->data;
  dim3 blockSize(1024);
  dim3 gridSize((num + blockSize.x - 1) / blockSize.x);
  array_set_kernel<<<gridSize, blockSize>>>(num, arr_data, value);
  cudaDeviceSynchronize();
  return 0;
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
  int in_ndim = input->ndim;
  int ex_ndim = output->ndim - input->ndim;
  int in_num = 1;
  int ex_num = 1;
  for (int i = 0; i < ex_ndim; ++i) {
    ex_num *= output->shape[i];
  }
  for (int i = 0; i < in_ndim; ++i) {
    assert(input->shape[i] == output->shape[i + ex_ndim]);
    in_num *= input->shape[i];
  }
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  dim3 blockSize(1024);
  dim3 gridSize((in_num + blockSize.x - 1) / blockSize.x);
  broadcast_to_kernel<<<gridSize, blockSize>>>(
    in_num, ex_num, input_data, output_data);
  cudaDeviceSynchronize();
  return 0;
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  assert(input->ndim == output->ndim + 1);
  int reduce_num = input->shape[0];
  int remain_num = 1;
  for (int i = 1; i < input->ndim; ++i) {
    remain_num *= input->shape[i];
  }
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  dim3 blockSize(1024);
  dim3 gridSize((remain_num + blockSize.x - 1) / blockSize.x);
  reduce_sum_axis_zero_kernel<<<gridSize, blockSize>>>(
    reduce_num, remain_num, input_data, output_data);
  cudaDeviceSynchronize();
  return 0;
}

int twoMatElemOp(const DLArrayHandle matA, const DLArrayHandle matB, 
                 DLArrayHandle output, ElemOp elemOp) {
  checkMatrixShape(matA, matB, output);
  int nrow = matA->shape[0];
  int ncol = matA->shape[1];
  const float *mat_a_data = (const float *)matA->data;
  const float *mat_b_data = (const float *)matB->data;
  float *output_data = (float *)output->data;
  dim3 blockSize(32, 32);
  dim3 gridSize((nrow + blockSize.x - 1) / blockSize.x, 
                (ncol + blockSize.y - 1) / blockSize.y);
  switch (elemOp) {
    case Add:
      matrix_elem_add_kernel<<<gridSize, blockSize>>>(
        nrow, ncol, mat_a_data, mat_b_data, output_data);   
      break;
    case Mul:
      matrix_elem_mul_kernel<<<gridSize, blockSize>>>(
        nrow, ncol, mat_a_data, mat_b_data, output_data);
      break;
    default:
      assert(false);
  }
  cudaDeviceSynchronize();
  return 0;
}

int oneMatElemOp(const DLArrayHandle input, DLArrayHandle output,
                 ElemOp elemOp, float val = 0) {
  checkMatrixShape(input, output);
  int nrow = input->shape[0];
  int ncol = input->shape[1];
  const float *input_data = (const float *)input->data;
  float *output_data = (float* )output->data;
  dim3 blockSize(32, 32);
  dim3 gridSize((nrow + blockSize.x - 1) / blockSize.x, 
                (ncol + blockSize.y - 1) / blockSize.y);
  switch (elemOp) {
    case AddByConst:
      matrix_elem_add_by_const_kernel<<<gridSize, blockSize>>>(
        nrow, ncol, input_data, val, output_data);
      break;
    case MulByConst:
      matrix_elem_mul_by_const_kernel<<<gridSize, blockSize>>>(
        nrow, ncol, input_data, val, output_data);
      break;
    default:
      assert(false);
  }
  cudaDeviceSynchronize();
  return 0;
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, 
                              DLArrayHandle output) {
  return twoMatElemOp(matA, matB, output, Add);
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
  return oneMatElemOp(input, output, AddByConst, val);
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  return twoMatElemOp(matA, matB, output, Mul);
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
  return oneMatElemOp(input, output, MulByConst, val);
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  // Hint: use cublas
  // cublas assume matrix is column major
  assert(matA->ndim == 2 && matB->ndim == 2);
  int rowA = matA->shape[0];
  int colA = matA->shape[1];
  int rowB = matB->shape[0];
  int colB = matB->shape[1];
  
  cublasHandle_t handle;
  cublasStatus_t status = cublasCreate(&handle);
  assert(status == CUBLAS_STATUS_SUCCESS);

  float alpha = 1.0;
  float beta = 0.0;
  
  int m = transposeB ? rowB : colB;
  int n = transposeA ? colA : rowA;
  int k = transposeA ? rowA : colA;

  cublasOperation_t trans_a = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t trans_b = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;

  int lda = colB; 
  int ldb = colA;
  int ldc = m;

  const float *data_a = (const float *)matB->data;
  const float *data_b = (const float *)matA->data;
  float *data_c = (float *)matC->data;

  status = cublasSgemm(handle, trans_a, trans_b, 
                       m, n, k, &alpha,
                       data_a, lda, data_b, ldb,
                       &beta, data_c, ldc);
  assert(status == CUBLAS_STATUS_SUCCESS);

  status = cublasDestroy(handle);
  assert(status == CUBLAS_STATUS_SUCCESS);
  return 0;
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  assert(input->ndim == output->ndim); 
  int ndim = input->ndim;
  int num = 1;
  for (int i = 0; i < ndim; ++i) {
    assert(input->shape[i] == output->shape[i]);
    num *= input->shape[i];
  }
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  dim3 blockSize(1024);
  dim3 gridSize((num + blockSize.x - 1) / blockSize.x);
  relu_kernel<<<gridSize, blockSize>>>(num, input_data, output_data);
  cudaDeviceSynchronize();
  return 0;
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  assert(input->ndim == in_grad->ndim && in_grad->ndim == output->ndim);
  int ndim = input->ndim;
  int num = 1;
  for (int i = 0; i < ndim; ++i) {
    assert(input->shape[i] == in_grad->shape[i] &&
           in_grad->shape[i] == output->shape[i]);
    num *= input->shape[i];
  }
  const float *input_data = (const float *)input->data;
  const float *in_grad_data = (const float *)in_grad->data;
  float *output_data = (float *)output->data;
  dim3 blockSize(1024);
  dim3 gridSize((num + blockSize.x - 1) / blockSize.x);
  relu_gradient_kernel<<<gridSize, blockSize>>>(
    num, input_data, in_grad_data, output_data);
  cudaDeviceSynchronize();
  return 0; 
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
  checkMatrixShape(input, output);
  int nrow = input->shape[0];
  int ncol = input->shape[1];
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  dim3 blockSize(1024);
  dim3 gridSize((nrow + blockSize.x - 1) / blockSize.x);
  softmax_kernel<<<gridSize, blockSize>>>(
    nrow, ncol, input_data, output_data);
  cudaDeviceSynchronize();
  return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
  checkMatrixShape(input_a, input_b);
  assert(output->ndim == 1);
  int nrow = input_a->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input_a->shape[1];
  const float *input_data_a = (const float *)input_a->data;
  const float *input_data_b = (const float *)input_b->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
    nrow, ncol, input_data_a, input_data_b, output_data);
  cudaDeviceSynchronize();
  return 0;
}
