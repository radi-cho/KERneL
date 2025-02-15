def llm_query(python_source, context):
    function_name = "diag_matmul_kernel"

    cpp = """
#include <torch/torch.h>

torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B) {
   int N = A.size(0);
   int M = B.size(1);
   auto C = torch::zeros({N, M}, A.options());

   int total_elements = N * M;
   const int threads_per_block = 256;
   int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

   diag_matmul_kernel<<<blocks, threads_per_block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M);

   return C;
}#"""
    cuda = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void diag_matmul_kernel(const float* A, const float* B, float* C, int N, int M) {
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   int total_elements = N * M;
   if (index < total_elements) {
       int i = index / M;
       int j = index % M;
       C[index] = A[i] * B[i * M + j];
   }
}#"""
    return function_name, cuda, cpp

def llm_query2(python_source, context):
    function_name = "diag_matmul_cuda"
    cuda = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void diag_matmul_kernel(
    const float* diag,
    const float* mat,
    float* out,
    const int N,
    const int M) {
    
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < M) {
        out[row * M + col] = diag[row] * mat[row * M + col];
    }
}

torch::Tensor diag_matmul_cuda(torch::Tensor diag, torch::Tensor mat) {
    const int N = diag.size(0);
    const int M = mat.size(1);
    
    auto out = torch::zeros({N, M}, mat.options());
    
    const dim3 threads(16, 16);
    const dim3 blocks((M + threads.x - 1) / threads.x,
                     (N + threads.y - 1) / threads.y);
                     
    diag_matmul_kernel<<<blocks, threads>>>(
        diag.data_ptr<float>(),
        mat.data_ptr<float>(),
        out.data_ptr<float>(),
        N, M);
        
    return out;
}
"""
    cpp = "torch::Tensor diag_matmul_cuda(torch::Tensor diag, torch::Tensor mat);"

    return function_name, cuda, cpp