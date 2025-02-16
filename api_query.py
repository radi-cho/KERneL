from openai import OpenAI
from typing import Union, Tuple, List
import os
from datetime import datetime
import concurrent.futures
import time
import random
from prompt_construction import prompt_generate_ex_with_CoT_template, prompt_fix_correctness
from utils import extract_method_name, save_reasoning, save_kernels, extract_from_text, initialize_client

PROMPT_PREFIX_PATH = "prompt_prefix.txt"
PROMPT_POSTFIX_PATH = "prompt_postfix.txt"
MAX_REASONING_TOKENS = 2000
MAX_EXTRACTION_TOKENS = 1000
NUM_SAMPLES = 1

DEBUG = True
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

USE_OPEN_AI = True

if USE_OPEN_AI:
    EXTRACTION_MODEL = "gpt-4o" 
    KERNEL_GEN_MODEL = "o1-2024-12-17" 
else:
    EXTRACTION_MODEL = "qwen/qwen2.5-7b-instruct" 
    KERNEL_GEN_MODEL = "deepseek-ai/deepseek-r1"

KERNEL_CU_CPP_FLAGS = [("<kernel_cu>", "</kernel_cu>"), ("<cpp_kernel>", "</cpp_kernel>")]
INIT_INPUT_FUNC_FLAGS = [("<model_initialization_code>", "</model_initialization_code>"), ("<input_function>", "</input_function>")]

NVIDIA_API_KEY = "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"

CLIENTS = []

model_init_code = None
get_input_function_code = None

def init_multiple_clients(API_type: str):
    if API_type == "OpenAI":
        USE_OPEN_AI = True
    elif API_type == "Deepseek":
        USE_OPEN_AI = False
    else:
        USE_OPEN_AI = False
    if len(CLIENTS) == 0:
        for i in range(NUM_SAMPLES):
            client = initialize_client(api_key = None, base_url = None)
            CLIENTS.append(client)
            print(f"Client {i} Initialized")

def query_kernel_generation( 
                 model_type: str, 
                 pytorch_function: str, 
                 additional_context: str = "",
                 stream = True) -> str:
    
    system_prompt = prompt_generate_ex_with_CoT_template(
        ref_arch_src = pytorch_function,
        cot_example = "ex_fuse_gelu")
        
    with open(PROMPT_PREFIX_PATH, 'r') as file:
        prompt_prefix = file.read()
    with open(PROMPT_POSTFIX_PATH, 'r') as file:
        prompt_postfix = file.read()

    if len(additional_context) == 0:
        system_prompt = f"{prompt_prefix} {system_prompt} {pytorch_function} {prompt_postfix}"
    else:
        system_prompt = f"{prompt_prefix} {system_prompt} {pytorch_function} Here is Feedback for Your Last Function {additional_context}  {prompt_postfix}"

    if USE_OPEN_AI:
        completion = CLIENTS[0].chat.completions.create(
            model=model_type,
            messages= [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Write a CUDA kernel for the following PyTorch function:\n{pytorch_function}"}
            ],
            stream=stream
        )
    else:
        completion = CLIENTS[0].chat.completions.create(
            model=model_type,
            messages= [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Write a CUDA kernel for the following PyTorch function:\n{pytorch_function}"}
            ],
            temperature=0.6,
            top_p=0.7,
            max_tokens=MAX_REASONING_TOKENS,
            stream=stream
        )

    reasoning_response = []
    for chunk in completion:
        content = chunk.choices[0].delta.content
        reasoning_response.append(content)

        if NUM_SAMPLES == 1 and DEBUG:
            print(content, end="")
 
    return reasoning_response

def query_extraction(response_text: str, 
                     refinement_client: OpenAI, 
                     model_type: str, 
                     system_prompt: str = None, 
                     stream: bool = True):
    
    if system_prompt is None:
        system_prompt = """
        You are given a text that contains CUDA wrapper code for kernel.cpp and kernel.cu. 
        Your task is to extract the relevant code and return it structured within XML-like tags, as shown below:

        <kernel_cu>
        <insert the complete CUDA code for kernel.cu here, including all helper macros, imports, and kernel function implementations>
        </kernel_cu>

        <cpp_kernel>
        <insert only the kernel.cpp method signature here, as a single-line declaration with the same name as the function defined in kernel.cu>
        </cpp_kernel>

        ### Rules:
        1. **Consistency**:
        - The function names in `<kernel_cu>` and `<cpp_kernel>` must match.
        - For example, if `kernel_cu` contains a function named `diag_matmul`, the `cpp_kernel` must include the corresponding signature: `torch::Tensor diag_matmul(torch::Tensor A, torch::Tensor B);`.

        2. **`<kernel_cu>`**:
        - Extract the **entire CUDA code**, including:
            - All includes (`#include`).
            - Macros or helper definitions (e.g., `CHECK_CUDA`, `CHECK_INPUT`).
            - All kernel implementations (`__global__ void` functions).
            - Any wrapper functions defined in `kernel.cu` (e.g., `diag_matmul` in the example).
        - Ensure the extracted code is complete and compilable.

        3. **`<cpp_kernel>`**:
        - Extract only the function signature for the CUDA wrapper.
        - Format the signature on a single line, e.g., `torch::Tensor diag_matmul(torch::Tensor A, torch::Tensor B);`.

        4. **No Extra Content**:
        - Do not include any comments, explanations, or text outside the specified tags.
        - Ensure the output ends cleanly without trailing characters or extra tokens.

        5. **Proper Escaping**:
        - Ensure all quotes and special characters in the extracted code are properly escaped to maintain valid syntax.

        ---

        ### Example Input:
        ```cpp
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>

        #define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
        #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
        #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

        __global__ void diag_matmul_kernel(
            float* output,
            const float* A,
            const float* B,
            const int N,
            const int M
        ) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            const int j = blockIdx.y * blockDim.y + threadIdx.y;

            if (i < N && j < M) {
                output[i * M + j] = A[i] * B[i * M + j];
            }
        }

        torch::Tensor diag_matmul(const torch::Tensor& A, const torch::Tensor& B) {
            CHECK_INPUT(A);
            CHECK_INPUT(B);
            
            const int N = A.size(0);
            const int M = B.size(1);
            auto output = torch::empty({N, M}, A.options());

            const dim3 threads(16, 16);
            const dim3 blocks(
                (N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y
            );

            diag_matmul_kernel<<<blocks, threads>>>(
                output.data_ptr<float>(),
                A.data_ptr<float>(),
                B.data_ptr<float>(),
                N,
                M
            );

            return output;
        }

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("diag_matmul", &diag_matmul, "Diagonal matrix multiplication (CUDA)");
        }

        """
    
    if USE_OPEN_AI:
        completion = refinement_client.chat.completions.create(
            model=model_type,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract the CUDA and C++ method signature (kernel.cu and kernel.cpp) from the following text:\n{response_text}"}
            ],
            stream=stream
        )
    else:
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
    """
    returns kernel_cu, kernel_cpp as str
    """
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
                model_type=model_type,
                pytorch_function=pytorch_function,
                additional_context=additional_context,
                stream=True
            )
        except Exception as e:
            wait_time = 2 ** attempt + random.uniform(0, 1)  # Exponential backoff with jitter
            print(f"{e}: Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
    raise Exception("Max retries exceeded for query.")  # Raise an exception if all retries fail

def generate_multiple_kernels(
    pytorch_function: str, 
    additional_context: str = "", 
    use_extraction_client: bool = True,
    API_type: str = "OpenAI",
    ) -> Union[List[Tuple[str, str, str]], Tuple[str, str, str]]:

    if len(CLIENTS) == 0:
        init_multiple_clients(API_type)

    generated_kernels = []
    processed_kernels = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(generate_single_kernel, CLIENTS[query_id], 
                            KERNEL_GEN_MODEL, pytorch_function, 
                            additional_context, True)
            for query_id in range(NUM_SAMPLES)
        ]
        for future in concurrent.futures.as_completed(futures):
            print("Generated Answer:")
            result = future.result()
            result = [item for item in result if item is not None]
            result = "".join(result)
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
                executor.submit(parse_reasoning_response, completion, CLIENTS[id], EXTRACTION_MODEL)
                for id, completion in enumerate(generated_kernels)
            ]
            for future in concurrent.futures.as_completed(process_futures):
                try:
                    cuda_kernel, cpp_kernel_signature = future.result()
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
        pytorch_function: str, 
        API_type: str = "OpenAI",
        model_type: str = EXTRACTION_MODEL, 
        stream = True) -> Tuple[str, str]:
    
    global model_init_code, get_input_function_code 

    if len(CLIENTS) == 0:
        init_multiple_clients(API_type)

    system_prompt = """
    You are an expert in analyzing PyTorch code. Your task is to identify the main PyTorch model defined in the given code, analyze its forward function, and write Python code that initializes the model and generates random inputs for its forward function. Follow these rules:

    ### Input:
    You will receive a Python script written in PyTorch. The script may contain:
    1. Multiple classes and model definitions.
    2. Standalone functions, including functions like `get_inputs`.

    ### Task:
    1. **Identify the Main PyTorch Model**:
    - If the code contains a class with a `forward` method, identify it as the main model.
    - Write code to initialize the model with reasonable default parameters. Assume default arguments for any constructor parameters unless explicitly defined in the code.

    2. **Handle Standalone Functions**:
    - If the provided code does **not** contain a model class but only a standalone function, wrap the function in a PyTorch model class and store the resulting model in a variable named `model`. Example:
        ```python
        class WrappedModel(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, *args, **kwargs):
                return <function_name>(*args, **kwargs)

        model = WrappedModel()
        ```

    3. **Handle Existing `get_inputs` Function**:
    - If the provided code contains a function named `get_inputs`, return an empty string (`""`) for the input function.

    4. **Generate Input Function**:
    - If the code does not contain a `get_inputs` function, analyze the `forward` method to determine the input arguments, their expected data types (`dtype`), and dimensions.
    - Write a Python function (`get_inputs`) that generates random tensors matching the expected `dtype` and dimensions.
    - IMPORTANT Use a power of 2 greater than 2^10 for any dimensions, such that the GPU can properly be utilized.

    ### Output Format:
    Return the output in the following structured XML-like format:

    ```plaintext
    <model_initialization_code>
    <insert the model initialization code here>
    </model_initialization_code>

    <input_function>
    <function that generates random inputs for the forward function, or an empty string if `get_inputs` already exists>
    </input_function>

    """

    if USE_OPEN_AI:
        completion = CLIENTS[0].chat.completions.create(
            model=model_type,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": f"Extract the model type, dtype, and dimensions of the input from the following PyTorch code:\n\n{pytorch_function.strip()}"}
            ],
            stream=stream
        )
    else:
        completion = CLIENTS[0].chat.completions.create(
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
    
    if "def get_inputs():" in pytorch_function:
        get_input_function_code = ""

    return model_init_code, get_input_function_code

def test_case(
    pytorch_function: str, 
    additional_context: str = "", 
    use_extraction_client: bool = True
    ):
    global model_init_code, get_input_function_code 

    if True:
        model_init, get_input_func = get_init_and_input_function(
            model_type = EXTRACTION_MODEL,
            pytorch_function = pytorch_function)
    else:
        model_init, get_input_func = "", ""
    
    print(f"model_init_code: \n{model_init}")
    print(f"get_input_func: \n{get_input_func}")

    model_context = f"The forward function you are creating is from {model_init_code} with input parameters sampled from {get_input_func}"
    additional_context = f"{model_context} {additional_context}"

    kernels = generate_multiple_kernels(
        pytorch_function = pytorch_function, 
        additional_context = additional_context, 
        use_extraction_client = use_extraction_client)
    
    print(f"func_name \n {kernels[0]} \n cuda_code: \n {kernels[1]} \n cpp_method_signature: \n {kernels[2]}")

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

    test_case(pytorch_function)
