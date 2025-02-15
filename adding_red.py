import streamlit as st
import requests
import random
from streamlit_ace import st_ace
import torch
import torch.nn as nn
from torchview import draw_graph
import io
from PIL import Image
import sys
import contextlib
from io import StringIO
import traceback

# Function to safely execute PyTorch code and get model
def execute_pytorch_code(code_string):
    try:
        # Create a dictionary for local variables
        local_dict = {}
        
        # Execute the code in the local context
        exec(code_string, {'torch': torch, 'nn': nn}, local_dict)
        
        # Find the model class (inherits from nn.Module)
        model_class = None
        for item in local_dict.values():
            if isinstance(item, type) and issubclass(item, nn.Module) and item != nn.Module:
                model_class = item
                break
                
        if model_class is None:
            return None, "No PyTorch model class found"
            
        # Initialize the model
        model = model_class()
        return model, None
        
    except Exception as e:
        return None, f"Error executing code: {str(e)}\n{traceback.format_exc()}"

# Function to create model visualization
def visualize_model(model, input_size):
    try:
        # Create model graph
        graph = draw_graph(
            model, 
            input_size=input_size,
            expand_nested=True,
            graph_dir='LR',  # Left to right direction
            hide_inner_tensors=False,
            hide_module_functions=False,
            roll=False
        )
        
        # Save graph to a file-like object
        graph.visual_graph.render('model_graph', format='png')
        
        # Read the generated image
        with open('model_graph.png', 'rb') as f:
            return f.read()
    except Exception as e:
        return None, f"Error generating visualization: {str(e)}"

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

# Model Visualization Section
st.markdown("🧠 **PyTorch Model Visualization**", unsafe_allow_html=True)
if python_code:
    try:
        # Convert input shape string to tuple
        try:
            input_size = tuple(map(int, input_shape.split(',')))
        except:
            st.error("Invalid input shape format. Please use comma-separated numbers.")
            input_size = None

        if input_size:
            # Execute the code and get model
            model, error = execute_pytorch_code(python_code)
            
            if error:
                st.error(error)
            elif model:
                st.success("✅ PyTorch model detected! Generating visualization...")
                
                # Create and display visualization
                graph_image = visualize_model(model, input_size)
                if isinstance(graph_image, bytes):
                    st.image(graph_image, use_column_width=True)
                    
                    # Display model summary
                    st.markdown("### Model Summary")
                    summary_output = StringIO()
                    with contextlib.redirect_stdout(summary_output):
                        total_params = sum(p.numel() for p in model.parameters())
                        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    
                    st.markdown(f"""
                    - Total Parameters: {total_params:,}
                    - Trainable Parameters: {trainable_params:,}
                    - Non-trainable Parameters: {total_params - trainable_params:,}
                    """)
                else:
                    st.error(f"Error generating visualization: {graph_image}")

    except Exception as e:
        st.error(f"Error processing model: {str(e)}")
        
# **🔹 Button to Transform Python Code (Future AI Model)**
st.markdown("⚙️ **Transform Python to CUDA Kernel**", unsafe_allow_html=True)
if st.button("🚀 Generate Kernel Code"):
    st.warning("⚠️ Model transformation is not implemented yet. This will call the AI model in the future.")

st.success("🚀 AI-driven kernel optimization is making computations **faster and smarter**! 🔥")

