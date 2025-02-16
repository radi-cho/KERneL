from openai import OpenAI
import json
from typing import Union, Tuple, List, Callable, Any
import os
import requests
from datetime import datetime
import concurrent.futures
import torch.nn as nn
import time
import random
from prompt_construction import prompt_generate_ex_with_CoT_template
from utils import extract_method_name, save_reasoning, save_kernels, extract_from_text, initialize_client

PROMPT_PREFIX_PATH = "prompt_prefix.txt"
PROMPT_POSTFIX_PATH = "prompt_postfix.txt"
MAX_REASONING_TOKENS = 2000
MAX_EXTRACTION_TOKENS = 1000
NUM_SAMPLES = 1

DEBUG = True
BASE_URL = "https://integrate.api.nvidia.com/v1"
EXTRACTION_MODEL = "qwen/qwen2.5-7b-instruct" #"meta/llama-3.2-3b-instruct"
DEEPSEEKR1_MODEL = "deepseek-ai/deepseek-r1"

KERNEL_CU_CPP_FLAGS = [("<kernel_cu>", "</kernel_cu>"), ("<cpp_kernel>", "</cpp_kernel>")]
INIT_INPUT_FUNC_FLAGS = [("<model_initialization_code>", "</model_initialization_code>"), ("<input_function>", "</input_function>")]

API_KEY = "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"

model_init_code = None
get_input_function_code = None

def query_kernel_generation(client: OpenAI, 
                 model_type: str, 
                 pytorch_function: str, 
                 additional_context: str = "",
                 stream = True) -> str:
    
    if False:
        with open(PROMPT_PREFIX_PATH, 'r') as file:
            prompt_prefix = file.read()
        with open(PROMPT_POSTFIX_PATH, 'r') as file:
            prompt_postfix = file.read()

        if len(additional_context) == 0:
            system_prompt = f"{prompt_prefix} {pytorch_function} {prompt_postfix}"
        else:
            system_prompt = f"{prompt_prefix} {pytorch_function} Here is Feedback for Your Last Function {additional_context}  {prompt_postfix}"
    else:
        system_prompt = prompt_generate_ex_with_CoT_template(
            ref_arch_src = pytorch_function,
            cot_example = "ex_fuse_gelu")
        
    completion = client.chat.completions.create(
        model=model_type,
        messages= [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Write a CUDA kernel for the following PyTorch function:\n{pytorch_function}"}
        ],
        temperature=0.6,
        top_p=0.7,
        max_tokens=MAX_REASONING_TOKENS,
        stream=stream
        #response_format={'type': 'json_object'} if response_format == "json" else None,
    )

    reasoning_response = []
    for chunk in completion:
        #print("Processing text chunk...")
        content = chunk.choices[0].delta.content
        reasoning_response.append(content)
 
    return reasoning_response

def query_extraction(response_text, refinement_client, model_type, system_prompt=None, stream=True):
    if system_prompt is None:
        system_prompt = """
        You are given a text that contains CUDA wrapper code for kernel.cpp and kernel.cu. 
        Your task is to extract the relevant code and return it structured within XML-like tags, as shown below:

        <kernel_cu>
        <insert the complete kernel.cu code here>
        </kernel_cu>

        <cpp_kernel>
        <insert only the kernel.cpp method signature here, as a single-line declaration with same name as kernel.cu>
        </cpp_kernel>

        Rules:
        0. IMPORTANT!!! Ensure the name of the functions in both are the same.
        3. For kernel.cpp, return only the function signature as a single line, e.g., `torch::Tensor function_name(torch::Tensor a, torch::Tensor b);`
        1. Do not include any text or comments outside of the specified tags.
        2. Ensure that each section is complete and valid, containing only the respective code for kernel.cpp and kernel.cu.
        
        4. The output should end cleanly, with no extra tokens, trailing characters, or explanations.
        5. Ensure all quotes and special characters within the code are properly escaped.

        Return only the structured output as described above. Do not include any additional text, comments, or explanations.
        """
       
    completion = refinement_client.chat.completions.create(
        model=model_type,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract the CUDA and C++ method signature (kernel.cu and kernel.cpp) from the following text:\n{response_text}"}
        ],
        temperature=0.6,
        top_p=0.7,
        max_tokens= MAX_EXTRACTION_TOKENS,  # Adjusted for larger responses
        stream=stream
    )

    return completion

def parse_reasoning_response(reasoning_text: str, 
                     refinement_client: OpenAI, 
                     refinement_type: str) -> Tuple[str, str]:
    
    extracted_answers = query_extraction(response_text = reasoning_text, 
                                   refinement_client = refinement_client, 
                                   model_type = refinement_type, 
                                   stream = True)

    cuda_response = []
    for chunk in extracted_answers:
        content = chunk.choices[0].delta.content
        if content:
            cuda_response.append(content)

    # Combine the response chunks into a full response string
    full_response = "".join(cuda_response).strip()

    print("\nFinished streaming response:")
    print(full_response)
    
    # Extract kernel.cpp and kernel.cu using XML-like tags
    return extract_from_text(full_response, KERNEL_CU_CPP_FLAGS)

def generate_single_kernel(client, model_type, pytorch_function, additional_context, stream, max_retries = 5):
    for attempt in range(max_retries):
        try:
            return query_kernel_generation(
                client=client,
                model_type=model_type,
                pytorch_function=pytorch_function,
                additional_context=additional_context,
                stream=True
            )
        except Exception as e:
            wait_time = 2 ** attempt + random.uniform(0, 1)  # Exponential backoff with jitter
            print(f"RateLimitError: Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
    raise Exception("Max retries exceeded for query.")  # Raise an exception if all retries fail

def generate_multiple_kernels(
    pytorch_function: str, 
    additional_context: str = "", 
    clients: List[OpenAI] = None,
    use_extraction_client: bool = True
    ) -> Union[List[Tuple[str, str, str]], Tuple[str, str, str]]:
    
    if clients is None:
        clients = []
        for i in range(NUM_SAMPLES):
            client = initialize_client(api_key = API_KEY, base_url = BASE_URL)
            clients.append(client)
            print(f"Client {i} Initialized")

    generated_kernels = []
    processed_kernels = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(generate_single_kernel, clients[query_id], 
                            DEEPSEEKR1_MODEL, pytorch_function, 
                            additional_context, True)
            for query_id in range(NUM_SAMPLES)
        ]
        for future in concurrent.futures.as_completed(futures):
            print(f"Generated Answer: ")
            result = "".join(future.result())
            print(f"{result}")
            generated_kernels.append(result)
            if not use_extraction_client:
                try:
                    cpp_kernel_signature, cuda_kernel = extract_from_text(result, flags = KERNEL_CU_CPP_FLAGS)
                    processed_kernels.append((cpp_kernel_signature, cuda_kernel))
                    print(f"Kernel processed and collected: {len(processed_kernels)}")
                except Exception as e:
                    print(f"Error processing a kernel: {e}")

    if use_extraction_client:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit processing tasks for each generated kernel
            process_futures = [
                executor.submit(parse_reasoning_response, completion, clients[id], EXTRACTION_MODEL)
                for id, completion in enumerate(generated_kernels)
            ]

            # Collect the processed kernels (cpp_kernel, cuda_kernel) as they complete
            for future in concurrent.futures.as_completed(process_futures):
                try:
                    cpp_kernel_signature, cuda_kernel = future.result()
                    method_name = extract_method_name(cpp_signature = cpp_kernel_signature)
                    processed_kernels.append((method_name, cuda_kernel, cpp_kernel_signature))
                    print(f"Kernel processed and collected: {len(processed_kernels)}")
                except Exception as e:
                    print(f"Error processing a kernel: {e}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"reasoning_results_{timestamp}.txt"

    save_reasoning(generated_kernels, filename=output_filename)

    if NUM_SAMPLES > 1:
        return processed_kernels
    else:
        return processed_kernels[0]

def get_init_and_input_function(
        client: OpenAI,
        model_type: str, 
        pytorch_function: str, 
        stream = True) -> Tuple[str, str]:
    global model_init_code, get_input_function_code 

    system_prompt = """
    You are an expert in analyzing PyTorch code. Your task is to identify the main PyTorch model defined in the given code, analyze its forward function, and write Python code that initializes the model and generates random inputs for its forward function. Follow these rules:

    ### Input:
    You will receive a Python script written in PyTorch. The script may contain multiple classes, functions, and model definitions.

    ### Task:
    1. Identify the main PyTorch model class based on the presence of a `forward` method.
    - The main model is usually the class containing the primary `forward` function used in training or inference.
    - Write code to initialize the model with reasonable default parameters. Assume default arguments for any constructor parameters unless explicitly defined in the code.

    2. Identify the inputs to the `forward` function:
    - Analyze the `forward` method to determine the input arguments, their expected data types (`dtype`), and dimensions.
    - Write a Python function (`get_inputs`) that generates random tensors matching the expected `dtype` and dimensions.

    3. If the model constructor requires specific initialization inputs, write a separate function (`get_init_inputs`) to generate these random inputs.

    ### Output Format:
    Return the output in the following structured XML-like format:

    ```plaintext
    <model_initialization_code>
    <insert the model initialization code here>
    </model_initialization_code>

    <input_function>
    <function that generates random inputs for the forward function>
    </input_function>

    """
       
    completion = client.chat.completions.create(
        model=model_type,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": f"Extract the model type, dtype, and dimensions of the input from the following PyTorch code:\n\n{pytorch_function.strip()}"}
        ],
        temperature=0.6,
        top_p=0.7,
        max_tokens = MAX_EXTRACTION_TOKENS,  # Adjusted for larger responses
        stream=stream
    )

    response = []
    for chunk in completion:
        content = chunk.choices[0].delta.content
        if content:
            response.append(content)

    # Combine the response chunks into a full response string
    full_response = "".join(response).strip()
    model_init_code, get_input_function_code = extract_from_text(
        full_response, 
        flags = INIT_INPUT_FUNC_FLAGS)
    
    return model_init_code, get_input_function_code


def main(
    pytorch_function: str, 
    additional_context: str = "", 
    use_extraction_client: bool = True
    ):
    global model_init_code, get_input_function_code 

    # Initialize client and query API
    clients = []
    for i in range(NUM_SAMPLES):
        client = initialize_client(api_key = API_KEY, base_url = BASE_URL)
        clients.append(client)
        print(f"Client {i} Initialized")

    model_init, get_input_func = get_init_and_input_function(
        client = clients[0],
        model_type = EXTRACTION_MODEL,
        pytorch_function = pytorch_function)
    
    print(f"model_init_code: \n{model_init_code}")
    print(f"model_init_code: \n{get_input_function_code}")

    model_context = f"The forward function you are creating is from {model_init} with input parameters sampled from {get_input_func}"
    additional_context = f"{model_context} {additional_context}"

    kernels = generate_multiple_kernels(
                            clients = clients, 
                            pytorch_function = pytorch_function, 
                            additional_context = additional_context, 
                            use_extraction_client = use_extraction_client)
    
    print(f"func_name {kernels[0]} \n cuda_code: {kernels[1]} \n cpp_method_signature {kernels[2]}")

    save_kernels(kernels if NUM_SAMPLES > 1 else [kernels], directory = "sample")

    return {"model_init_func": model_init, "get_input_func": get_input_func, "kernels": kernels}

if __name__ == '__main__':
    pytorch_function = """
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
        
        def forward(self, A, B):
            return torch.diag(A) @ B

    M = 4096
    N = 4096

    def get_inputs():
        A = torch.randn(N)
        B = torch.randn(N, M)
        return [A, B]

    def get_init_inputs():
        return []  # No special initialization inputs needed
    """

    main(pytorch_function)

def debug():

    deepseek_output = """
    <think>
    Okay, I need to create a CUDA kernel to replace the PyTorch element-wise addition function. Let me start by understanding what the original function does. It takes two tensors a and b and returns their element-wise sum. The goal is to write a custom CUDA kernel that performs this operation more efficiently.

    First, I'll outline the steps required. The CUDA kernel should take two input tensors, add each corresponding element, and store the result in an output tensor. I'll need to write the kernel function itself, the launch parameters (block size, grid size), and the PyTorch wrapper to integrate it.

    I remember that in CUDA, each thread handles one element or a chunk of elements. For simplicity, I'll start with one thread per element. The block size can be 256 threads, which is a common choice. The grid size will then be the total number of elements divided by the block size, rounded up.

    Next, I need to handle the kernel launch. The inputs are PyTorch tensors, so I have to access their data pointers. Also, I need to ensure that the tensors are on the GPU and contiguous. The output tensor should be initialized with zeros_like(a) to match the shape and device.

    Wait, what about different tensor shapes? The original PyTorch function broadcasts if necessary, but for simplicity, maybe I'll assume that a and b are the same shape. The user's example uses a and b of the same shape, so perhaps that's acceptable here.

    Now, writing the CUDA kernel. The kernel will have parameters for the input pointers, output pointer, and the total number of elements. Each thread calculates its global index and checks if it's within bounds before performing the addition.

    Then, the PyTorch extension. Using torch.utils.cpp_extension.load_inline to compile the CUDA code. The cpp_sources and cuda_sources need to be properly formatted. The function name in the C++ code should be exposed so that Python can call it.

    Testing the code: I need to make sure that the ModelNew class uses the custom CUDA kernel instead of the default addition. The forward method will call the compiled function. Also, the get_inputs function should generate tensors on the GPU.

    Potential issues: Forgetting to check the tensor dimensions, incorrect grid/block calculations, or memory access issues. Also, ensuring that the CUDA code is correctly formatted with proper includes and syntax.

    Let me put it all together. The CUDA kernel is defined in a string, then loaded as a PyTorch extension. The ModelNew class uses this extension in its forward pass. The input tensors are created on the GPU to match the kernel's expectations.

    I should also verify that the code compiles and runs without errors. Using load_inline with verbose=True might help catch any compilation issues. The block and grid dimensions are calculated correctly to cover all elements.

    Finally, the get_inputs function in the example uses tensors of size 4096, which is a good test size. The user's original example had 1x128 tensors, but larger sizes would better demonstrate the benefits of parallelization.
    </think>

    ```python
    import torch
    from torch.utils.cpp_extension import load_inline

    # Custom CUDA kernel implementation for element-wise addition
    cuda_source = '''
    #include <torch/extension.h>
    #include <cuda.h>
    #include <cuda_runtime.h>

    __global__ void elementwise_add_kernel(
        const float* a,
        const float* b,
        float* out,
        int num_elements
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_elements) {
            out[idx] = a[idx] + b[idx];
        }
    }

    torch::Tensor custom_add_cuda(torch::Tensor a, torch::Tensor b) {
        // Validate inputs
        TORCH_CHECK(a.device().is_cuda(), "Input tensor a must be on GPU");
        TORCH_CHECK(b.device().is_cuda(), "Input tensor b must be on GPU");
        TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same shape");
        
        auto out = torch::zeros_like(a);
        const int num_elements = a.numel();
        
        // Kernel launch parameters
        const int threads_per_block = 256;
        const int blocks = (num_elements + threads_per_block - 1) / threads_per_block;
        
        // Ensure tensors are contiguous
        auto a_cont = a.contiguous();
        auto b_cont = b.contiguous();
        
        elementwise_add_kernel<<<blocks, threads_per_block>>>(
            a_cont.data_ptr<float>(),
            b_cont.data_ptr<float>(),
            out.data_ptr<float>(),
            num_elements
        );
        
        return out;
    }
    '''

    cpp_source = "torch::Tensor custom_add_cuda(torch::Tensor a, torch::Tensor b);"

    # Load the custom CUDA extension
    custom_add = load_inline(
        name='custom_add',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['custom_add_cuda'],
        verbose=True
    )

    class ModelNew(torch.nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, a, b):
            return custom_add.custom_add_cuda(a, b)

    def get_inputs():
        a = torch.randn(4096, device='cuda')  # Larger tensor for meaningful GPU utilization
        b = torch.randn(4096, device='cuda')
        return [a, b]

    def get_init_inputs():
        return []
    ```

    Key optimizations and features:
    1. **Batched element-wise operations**: Processes 4096 elements for better GPU utilization
    2. **Optimal thread configuration**: Uses 256 threads per block (common best practice)
    3. **Memory coalescing**: Ensures contiguous memory access patterns
    4. **Input validation**: Checks tensor devices and shapes match
    5. **Asynchronous execution**: Leverages CUDA's parallel execution model
    6. **Zero-copy tensor handling**: Uses PyTorch's memory management directly

    To use:
    ```python
    model = ModelNew().cuda()
    a, b = get_inputs()
    result = model(a, b)
    ```

    This implementation should provide significant speedups over the native PyTorch implementation for large tensors due to parallel execution on GPU cores
    """


    if False:
        deepseek_client = initialize_client(api_key = API_KEY, base_url = BASE_URL)
        print("Client Initialized")
        completion = query_kernel_generation(client = deepseek_client, 
                            model_type = DEEPSEEKR1_MODEL, 
                            pytorch_function = pytorch_function, 
                            response_format= "json", 
                            stream=True) 
        print("Response Found")
    else:
        completion = deepseek_output