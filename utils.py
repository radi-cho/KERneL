import re
from typing import Union, Tuple, List
import os
import time
from datetime import datetime
from openai import OpenAI

API_KEY = "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"


def initialize_client(api_key = API_KEY, base_url="https://integrate.api.nvidia.com/v1"):
    return OpenAI(
        base_url = base_url,
        api_key = api_key
    )

def extract_method_name(cpp_signature: str) -> str:
    """
    Extracts the method name from a C++ function signature.

    Args:
        cpp_signature (str): The C++ function signature as a string.
    
    Returns:
        str: The method name, or an empty string if not found.
    """
    # Use a regular expression to find the method name
    match = re.search(r'[\w:]+::(\w+)\s*\(', cpp_signature)
    if match:
        return match.group(1)  # Group 1 is the method name
    else:
        # If the namespace (::) is absent, handle it separately
        match = re.search(r'(\w+)\s*\(', cpp_signature)
        if match:
            return match.group(1)  # Group 1 is the method name
    return ""  # Return an empty string if no match is found


def extract_from_text(full_response: str, flags: List[Tuple[str, str]]) -> Tuple[str, str, str]:
    try:
        results = []
        for idx, (flag_start, flag_stop) in enumerate(flags):
            idx_start = full_response.find(flag_start) + len(flag_start)
            idx_end = full_response.find(flag_stop)
            results.append(full_response[idx_start:idx_end].strip() if idx_start != -1 and idx_end != -1 else "")
        return tuple(results)
    except Exception as e:
        print(f"Error parsing response: {e}")
        return tuple([None for _ in flags])


def save_reasoning(results: List[str], filename: str = "results.txt"):
    """
    Saves the generated results to a specified file.

    Args:
        results (List[str]): The list of string results to save.
        filename (str): The filename where results will be saved.
    """
    # Ensure the output directory exists
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, "w") as file:
        for idx, result in enumerate(results, start=1):
            file.write(f"Result {idx}:\n")
            file.write(f"{result}\n")
            file.write("-" * 50 + "\n")
    print(f"Results saved to {filename}")

def save_kernels(kernels: List[Tuple[str, str, str]], directory="sample"):
    
    # Get the current date and time for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = f"{directory}/{timestamp}"
    # Ensure the output directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    for idx, (method_name, cuda_kernel, cpp_kernel_signature) in enumerate(kernels):
        cpp_filename = os.path.join(directory, f"/kernel_{idx}.cpp")
        cu_filename = os.path.join(directory, f"kernel_{idx}.cu")

        with open(cpp_filename, "w") as cpp_file:
            cpp_file.write(cpp_kernel_signature)
        print(f"Saved C++ kernel to {cpp_filename}")

        # Save the cuda_kernel to a file
        with open(cu_filename, "w") as cu_file:
            cu_file.write(cuda_kernel)
        print(f"Saved CUDA kernel to {cu_filename}")
