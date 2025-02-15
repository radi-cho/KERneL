import streamlit as st
import requests
import random
from streamlit_ace import st_ace
import re
import graphviz

# Function to parse PyTorch model architecture
def parse_pytorch_code(code):
    try:
        # Look for class definition that inherits from nn.Module
        class_match = re.search(r'class\s+(\w+)\(.*?nn\.Module.*?\):(.*?)(?=\n\n|\Z)', code, re.DOTALL)
        if not class_match:
            return None
        
        # Extract the forward method
        forward_match = re.search(r'def\s+forward\s*\([^)]*\):(.*?)(?=\n\n|\Z)', code, re.DOTALL)
        if not forward_match:
            return None
            
        # Extract layer definitions
        layers = []
        layer_pattern = r'nn\.(Linear|Conv2d|MaxPool2d|ReLU|Flatten)\((.*?)\)'
        for line in code.split('\n'):
            if 'nn.' in line:
                layer_match = re.search(layer_pattern, line)
                if layer_match:
                    layer_type = layer_match.group(1)
                    params = layer_match.group(2)
                    layers.append((layer_type, params))
        
        return layers
    except Exception as e:
        st.error(f"Error parsing PyTorch code: {str(e)}")
        return None

# Function to create network visualization
def visualize_network(layers):
    dot = graphviz.Digraph(comment='Neural Network Architecture')
    dot.attr(rankdir='LR')  # Left to right layout
    
    # Node styling
    dot.attr('node', shape='box', style='rounded,filled', color='lightblue')
    
    # Add input node
    dot.node('input', 'Input')
    prev_node = 'input'
    
    # Add layer nodes
    for idx, (layer_type, params) in enumerate(layers):
        node_id = f'layer_{idx}'
        
        # Create label based on layer type
        if layer_type == 'Linear':
            label = f'Linear\n{params}'
        elif layer_type == 'Conv2d':
            label = f'Conv2D\n{params}'
        elif layer_type == 'MaxPool2d':
            label = f'MaxPool2D\n{params}'
        elif layer_type == 'ReLU':
            label = 'ReLU'
        elif layer_type == 'Flatten':
            label = 'Flatten'
        
        # Set color based on layer type
        if layer_type == 'Linear':
            dot.attr('node', fillcolor='lightblue')
        elif layer_type in ['Conv2d', 'MaxPool2d']:
            dot.attr('node', fillcolor='lightgreen')
        elif layer_type == 'ReLU':
            dot.attr('node', fillcolor='lightyellow')
        elif layer_type == 'Flatten':
            dot.attr('node', fillcolor='lightgray')
        
        dot.node(node_id, label)
        dot.edge(prev_node, node_id)
        prev_node = node_id
    
    return dot

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

# Neural Network Visualization Section
st.markdown("🧠 **Neural Network Architecture Visualization**", unsafe_allow_html=True)
if python_code:
    # Check if code contains PyTorch neural network
    if 'nn.Module' in python_code:
        layers = parse_pytorch_code(python_code)
        if layers:
            st.success("✅ PyTorch neural network detected! Generating visualization...")
            dot = visualize_network(layers)
            st.graphviz_chart(dot)
            
            # Display network summary
            st.markdown("### Network Summary")
            for idx, (layer_type, params) in enumerate(layers, 1):
                st.write(f"{idx}. **{layer_type}**: {params}")
        else:
            st.info("ℹ️ No valid PyTorch neural network structure found in the code.")

# **🔹 Button to Transform Python Code (Future AI Model)**
st.markdown("⚙️ **Transform Python to CUDA Kernel**", unsafe_allow_html=True)
if st.button("🚀 Generate Kernel Code"):
    st.warning("⚠️ Model transformation is not implemented yet. This will call the AI model in the future.")

st.success("🚀 AI-driven kernel optimization is making computations **faster and smarter**! 🔥")

