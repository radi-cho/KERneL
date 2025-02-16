import streamlit as st
import requests
import random
from streamlit_ace import st_ace

import torch
import torch.nn as nn
from torchviz import make_dot
import ast
import inspect
import tempfile

# Function to extract the forward method's argument names
def get_forward_args(model):
    forward_fn = model.forward
    sig = inspect.signature(forward_fn)
    return list(sig.parameters.keys())

# Function to parse the user-provided code and extract the model class
def parse_model_class(user_code):
    module = ast.parse(user_code)
    class_def = None
    for node in module.body:
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if (isinstance(base, ast.Attribute) and base.attr == 'Module') or \
                   (isinstance(base, ast.Name) and base.id == 'Module'):
                    class_def = node
                    break
    return class_def

# Streamlit UI Setup - Remove Top Blank Space
st.set_page_config(page_title="Python to CUDA Kernel Optimization", layout="wide")

# **🔹 Two-Column Layout (Editors at the Top)**
col1, col2 = st.columns(2)

# **🔹 Left Side: Python Code Input (Editable, Starts at the Top)**
with col1:
    st.markdown("✏️ **Enter Python Code**", unsafe_allow_html=True)
    python_code = st_ace(
        language="python",  # Python Syntax Highlighting
        theme="monokai",  # Dark theme
        placeholder="Write or paste your Python code here...",
        height=400,  # **Slightly reduced height**
        key="python_code_editor"
    )

    # **Hardware Selection & Optimization Duration (Always Visible)**
    st.markdown("⚙️ **Optimization Settings**", unsafe_allow_html=True)
    hardware = st.selectbox("💻 Select Hardware", ["NVIDIA H100", "NVIDIA A100"])
    optimization_time = st.slider("⏳ Optimization Duration (mins)", 1, 15, 5)

# **🔹 Right Side: CUDA Kernel Code Output (Read-Only, Starts at the Top)**
with col2:
    st.markdown("⚡ **Generated CUDA Kernel Code**", unsafe_allow_html=True)

    # **Placeholder CUDA Kernel Code**
    cuda_kernel_pseudo = """ 
    __global__ void kernel_function(float *input, float *output, int N) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < N) {
            output[idx] = input[idx] * input[idx];  // Example computation
        }
    }
    """

    # ✅ Syntax-Highlighted **Read-Only** CUDA Output
    st_ace(
        value=cuda_kernel_pseudo,
        language="c_cpp",  # Use C++ mode since CUDA mode isn't available
        theme="monokai",
        readonly=True,  # **Ensures it's not editable**
        height=400,  # **Slightly reduced height**
        key="cuda_code_output"
    )

    # **Live Updating Performance Graph (Always Visible)**
    st.markdown("📊 **Live Performance Graph**", unsafe_allow_html=True)

    # Generate random initial data
    random_values = [round(random.uniform(1.0, 2.5), 2) for _ in range(7)]
    constant_values = [2.0] * 7  # Constant red line at value 2.0

    # Chart.js Script
    chart_html = f"""
    <canvas id="performanceChart"></canvas>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        var ctx = document.getElementById('performanceChart').getContext('2d');
        var chartData = {{
            labels: [1, 2, 3, 4, 5, 10, 15],
            datasets: [
                {{
                    label: 'Speedup Factor (x)',
                    data: {random_values},
                    borderColor: 'rgba(66, 197, 245, 1)',
                    backgroundColor: 'rgba(66, 197, 245, 0.2)',
                    borderWidth: 2,
                    fill: true
                }},
                {{
                    label: 'Constant Baseline',
                    data: {constant_values},
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 2,
                    fill: false,
                    borderDash: [5, 5]  // Dashed line for distinction
                }}
            ]
        }};

        var performanceChart = new Chart(ctx, {{
            type: 'line',
            data: chartData,
            options: {{
                responsive: true,
                animation: {{
                    duration: 1000
                }},
                scales: {{
                    x: {{ title: {{ display: true, text: 'Optimization Time (mins)' }} }},
                    y: {{ title: {{ display: true, text: 'Speedup Factor (x)' }}, beginAtZero: false }}
                }}
            }}
        }});

        function updateChart() {{
            let newVal = Math.max(1.0, chartData.datasets[0].data[chartData.datasets[0].data.length - 1] + (Math.random() * 0.5 - 0.25)).toFixed(2);
            chartData.datasets[0].data.push(newVal);
            chartData.labels.push(chartData.labels[chartData.labels.length - 1] + 1);
            chartData.datasets[1].data.push(2.0); // Keep red line constant

            if (chartData.datasets[0].data.length > 10) {{
                chartData.datasets[0].data.shift();
                chartData.labels.shift();
                chartData.datasets[1].data.shift(); // Keep red line aligned
            }}
            performanceChart.update();
        }}

        setInterval(updateChart, 2000);
    </script>
    """

    st.components.v1.html(chart_html, height=250)

# Visualize button
if st.button("Visualize Model"):
    local_vars = {}
    try:
        # Execute the user-provided code
        exec(python_code, globals(), local_vars)

        # Retrieve the model instance
        model = None
        for var in local_vars.values():
            if isinstance(var, nn.Module):
                model = var
                break

        if model is None:
            st.error("No valid PyTorch model found in the provided code.")
        else:
            # Extract the forward method's argument names
            forward_args = get_forward_args(model)
            if len(forward_args) < 2:
                st.error("The forward method should have at least one input argument.")
            else:
                input_arg = forward_args[1]  # The first argument after 'self'
                # Prompt user for input tensor dimensions
                input_shape = st.text_input(f"Enter the shape of the input tensor for '{input_arg}' (e.g., 1, 3, 224, 224):")
                if input_shape:
                    try:
                        # Convert input shape to a tuple of integers
                        input_shape = tuple(map(int, input_shape.split(',')))
                        # Create a sample input tensor with the specified shape
                        sample_input = torch.randn(input_shape)
                        # Generate the computational graph
                        output = model(sample_input)
                        graph = make_dot(output, params=dict(model.named_parameters()))

                        # Render the graph to a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                            graph.render(tmpfile.name, format='png')
                            tmpfile.seek(0)
                            image_data = tmpfile.read()

                        # Display the image
                        st.image(image_data, caption="Model Computational Graph", use_column_width=True)
                    except Exception as e:
                        st.error(f"An error occurred while processing the input shape: {e}")
    
    except Exception as e:
        st.error(f"Error: {e}")
        
# **🔹 Button to Transform Python Code (Future AI Model)**
st.markdown("⚙️ **Transform Python to CUDA Kernel**", unsafe_allow_html=True)
if st.button("🚀 Generate Kernel Code"):
    st.warning("⚠️ Model transformation is not implemented yet. This will call the AI model in the future.")

st.success("🚀 AI-driven kernel optimization is making computations **faster and smarter**! 🔥")
