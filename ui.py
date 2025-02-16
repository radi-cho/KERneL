import streamlit as st
import requests
import random
import types

from streamlit_ace import st_ace

import time
import torch
from torch import nn

# Streamlit UI Setup - Remove Top Blank Space
st.set_page_config(page_title="Python to CUDA Kernel Optimization", layout="wide")

# 🔹 Two-Column Layout (Editors at the Top)
col1, col2 = st.columns(2)

# 🔹 Left Side: Python Code Input (Editable, Starts at the Top)
with col1:
    st.markdown("✏️ **Enter Python Code**", unsafe_allow_html=True)
    python_code = st_ace(
        language="python",
        theme="monokai",
        placeholder="Write or paste your Python code here...",
        height=400,
        key="python_code_editor"
    )

    # **Tensor Input Dimension Section**
    # st.markdown("📏 **Specify Tensor Input Dimensions**", unsafe_allow_html=True)
    # tensor_dim = st.text_input("Enter tensor dimensions (e.g., 1, 3, 224, 224)", "1, 3, 224, 224")

    # 🔹 Hardware Selection & Optimization Duration
    st.markdown("⚙️ **Optimization Settings**", unsafe_allow_html=True)
    hardware = st.selectbox("💻 Select Hardware", ["NVIDIA H100"])
    num_trials = st.slider("🔁 Number of Trials", 10, 200, 100)

    torch_output = st.empty()
    cuda_output = st.empty()


# 🔹 Right Side: CUDA Kernel Code Output (Read-Only, Starts at the Top)
with col2:
    st.markdown("⚡ **Generated CUDA Kernel Code**", unsafe_allow_html=True)

    # Placeholder for CUDA Kernel Code
    cuda_code_container = st.empty()

    # 🔹 Performance Metrics (Below CUDA Code)
    st.markdown("📊 **Performance Metrics**", unsafe_allow_html=True)
    torch_time_text = st.empty()
    kernel_time_text = st.empty()

    # from torchview import draw_graph  # ensure torchview is installed

    # st.markdown("🖼 **Model Computational Graph**", unsafe_allow_html=True)
    # if st.button("🖥 Generate Computational Graph"):
    #     # Show a loading message for exactly 3 seconds.
    #     with st.spinner("Loading computational graph..."):
    #         time.sleep(2)

    #     # 1. Parse tensor dimensions and create dummy input.
    #     try:
    #         input_dims = tuple(map(int, tensor_dim.split(',')))
    #         dummy_input = torch.rand(input_dims)
    #     except Exception as e:
    #         st.error(f"Error parsing tensor dimensions: {e}")
    #         st.stop()

    #     # 2. Execute the user code in a fresh module namespace.
    #     user_module = types.ModuleType("user_module")
    #     try:
    #         exec(compile(python_code, "<string>", "exec"), user_module.__dict__)
    #     except Exception as e:
    #         st.error(f"Error executing user code: {e}")
    #         st.stop()

    #     # 3. Extract the model (the first nn.Module instance found)
    #     model = next((v for k, v in user_module.__dict__.items() if isinstance(v, nn.Module)), None)
    #     if model is None:
    #         st.error("No valid nn.Module instance found in the provided code.")
    #         st.write("Module keys:", list(user_module.__dict__.keys()))
    #         st.stop()

    #     try:
    #         # 4. Use TorchView to generate the computational graph.
    #         # draw_graph returns a path to an image file by default,
    #         # so we can then display that image using st.image.

    #         graph_path = draw_graph(model, input_size=input_dims, roll=True)
    #         st.write(graph_path.visual_graph)
    #     except Exception as e:
    #         st.error(f"TorchView error: {e}")
    #         st.exception(e)

# 🔹 Button to Send Python Code
st.markdown("⚙️ **Transform Python to CUDA Kernel**", unsafe_allow_html=True)

if st.button("🚀 Generate kernel"):
    if python_code.strip():
        st.info("📡 Initializing task on server...")

        try:
            # 🔹 Step 1: Send Python Code to Initialize Task
            payload = {"python_source": python_code, "num_trials": num_trials}
            response = requests.post(
                "https://d4fa-209-20-157-139.ngrok-free.app/initialize_task",
                json=payload
            )

            if response.status_code == 200:
                data = response.json()
                task_id = data.get("task_id")
                torch_time = data.get("torch_time", "N/A")

                torch_time_text.markdown(f"🔥 **Torch Execution Time:** `{torch_time} ms`")
                torch_output.markdown("🔥 **Torch output sample:** \n```\n" + data["output"] + "\n```")

                if task_id:
                    st.info("📡 Initializing CUDA kernel...")

                    # 🔹 Step 2: Request CUDA Kernel Initialization
                    kernel_payload = {"task_id": task_id, "num_trials": num_trials}
                    kernel_response = requests.post(
                        "https://d4fa-209-20-157-139.ngrok-free.app/get_kernel",
                        json=kernel_payload
                    )

                    if kernel_response.status_code == 200:
                        kernel_data = kernel_response.json()
                        kernel_time = kernel_data.get("kernel_time", "N/A")

                        kernel_time_text.markdown(f"⚡ **CUDA Execution Time:** `{kernel_time} ms`")

                        if "kernel_code" in kernel_data:
                            # ✅ Render CUDA Kernel Code dynamically
                            cuda_code_container.markdown("```cpp\n" + kernel_data["kernel_code"] + "\n```")
                            cuda_output.markdown("**CUDA output sample:** \n```\n" + kernel_data["output"] + "\n```")
                            st.success("✅ CUDA kernel compiled successfully!")
                        else:
                            st.warning("⚠️ CUDA kernel not received.")
                    else:
                        st.error(f"❌ Kernel error: {kernel_response.text}")
                else:
                    st.error("❌ Task ID missing from response.")
            else:
                st.error(f"❌ Error from server: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Failed to connect to server: {e}")

    else:
        st.warning("⚠️ Please enter Python code before applying.")
