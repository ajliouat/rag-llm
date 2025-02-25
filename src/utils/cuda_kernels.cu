#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * 2.0f;  // Example operation
    }
}

torch::Tensor custom_op(torch::Tensor input) {
    auto output = torch::zeros_like(input);
    int size = input.numel();
    int threads = 1024;
    int blocks = (size + threads - 1) / threads;

    custom_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_op", &custom_op, "Custom CUDA kernel");
}