import sys
import torch
import torch.nn as nn
import importlib.util
from uuid import uuid4
from flask import Flask, request, jsonify
from torch.utils.cpp_extension import load_inline
from api_query import generate_multiple_kernels, get_init_and_input_function
from llm_query_sample import llm_query

TASKS = {}


def initialize_python_module(source, module_name="dynamic_kernel_module"):
    try:
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        dynamic_module = importlib.util.module_from_spec(spec)
        exec(source, dynamic_module.__dict__)
        sys.modules[module_name] = dynamic_module
        return True, dynamic_module
    except Exception as e:
        return False, str(e)


def initialize_kernel_module(cuda_sources, cpp_sources, function_name, kernel_name="dynamic_cuda_kernel"):
    try:
        kernel_module = load_inline(
            name=kernel_name,
            cuda_sources=cuda_sources,
            cpp_sources=cpp_sources,
            functions=[function_name],
            verbose=True
        )

        return True, kernel_module
    except Exception as e:
        return False, str(e)


def time_execution_with_cuda_event(model, inputs, num_trials):
    device = torch.cuda.current_device()
    if isinstance(model, torch.nn.Module):
        model.to(device=device)
    inputs = [inp.to(device=device) for inp in inputs]

    elapsed_times = 0
    for trial in range(num_trials):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        if trial == num_trials - 1:
            output = model(*inputs)
        else:
            model(*inputs)

        end_event.record()

        torch.cuda.synchronize(device=device)
        elapsed_time_ms = start_event.elapsed_time(end_event)
        elapsed_times += elapsed_time_ms

    return elapsed_times / num_trials, output


app = Flask(__name__)

@app.route('/initialize_task', methods=['POST'])
def initialize_task():
    try:
        data = request.get_json()
        python_source = data.get("python_source", "")
        num_trials = data.get("num_trials", 100)

        model_init_code, get_input_function_code = get_init_and_input_function(python_source)
        imports = """
import torch
import torch.nn as nn
import numpy as np
"""
        python_source = imports + "\n\n" + python_source + "\n\n" + model_init_code + "\n" + get_input_function_code

        initialized, result = initialize_python_module(python_source)
        if initialized:
            model, get_inputs = getattr(result, "model"), getattr(result, "get_inputs")
            inputs = get_inputs()

            task_id = str(uuid4())
            average_time, output = time_execution_with_cuda_event(model, inputs, num_trials)
            TASKS[task_id] = [python_source, model, inputs, [["", average_time, output]]]
            response = {
                "status": "Task initialized successfully",
                "torch_time": average_time,
                "task_id": task_id,
                "output": str(output)
            }
            return jsonify(response), 200
    except Exception as e:
        response = {"error": str(e)}
        return jsonify(response), 400


@app.route('/get_kernel', methods=['POST'])
def get_kernel():
    try:
        data = request.get_json()

        num_trials = data.get("num_trials", 100)
        task_id = data.get("task_id", "")
        source, _, inputs, history = TASKS[task_id]

        additional_context = ""
        if len(history) > 1:
            additional_context = f"Your previous attempt with source: {history[-1][0]}\n\n runs in time {history[-1][1]} compared to a native PyTorch compilation which runs in time {history[0][1]}."

        function_name, cuda_sources, cpp_sources = generate_multiple_kernels(pytorch_function=source, additional_context=additional_context)
        initialized, result = initialize_kernel_module(cuda_sources, cpp_sources, function_name)

        if initialized:
            model = getattr(result, function_name)
            average_time, output = time_execution_with_cuda_event(model, inputs, num_trials)

            response = {
                "task_id": task_id,
                "status": "Kernel compiled successfully",
                "kernel_code": cuda_sources,
                "kernel_time": average_time,
                "output": str(output)
            }

            TASKS[task_id][3].append([cuda_sources, average_time, output])
            return jsonify(response), 200
        else:
            response = {
                "task_id": task_id,
                "status": "Failed during kernel compilation",
                "error": result
            }
            return jsonify(response), 200

    except Exception as e:
        return jsonify({
            "task_id": task_id,
            "error": str(e)
        }), 400


if __name__ == '__main__':
    app.run(debug=False)
