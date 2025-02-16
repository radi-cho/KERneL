source = """
<kernel_cu>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void max_pool2d_kernel(float* input, float* output, int channels, int input_height, int input_width, int pool_height, int pool_width, int stride, int output_height, int output_width) {
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (w_out < output_width && h_out < output_height) {
        float max_val = -FLT_MAX;
        int start_h = h_out * stride;
        int start_w = w_out * stride;

        for (int h = 0; h < pool_height; ++h) {
            for (int w = 0; w < pool_width; ++w) {
                int h_in = start_h + h;
                int w_in = start_w + w;
                if (h_in < input_height && w_in < input_width) {
                    max_val = fmaxf(max_val, input[c * input_height * input_width + h_in * input_width + w_in]);
                }
            }
        }
        output[c * output_height * output_width + h_out * output_width + w_out] = max_val;
    }
}

torch::Tensor max_pool2d_cuda(torch::Tensor input, int kernel_size, int stride) {
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    const int output_height = (input_height - kernel_size) / stride + 1;
    const int output_width = (input_width - kernel_size) / stride + 1;

    auto output = torch::empty({input.size(0), channels, output_height, output_width}, input.options());

    dim3 block_size(16, 16);
    dim3 num_blocks((output_width + block_size.x - 1) / block_size.x, (output_height + block_size.y - 1) / block_size.y, channels);

    max_pool2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        channels, 
        input_height, 
        input_width, 
        kernel_size, 
        kernel_size, 
        stride, 
        output_height, 
        output_width
    );

    return output;
}
</kernel_cu>
"""
cpp_src = """
<cpp_kernel>
torch::Tensor max_pool2d_cuda(torch::Tensor input, int kernel_size, int stride);
</cpp_kernel>
"""


