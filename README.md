# CUDA Kernel Generation with PyTorch

Welcome to our project! This repository provides examples and tools to help you explore and generate efficient CUDA kernels using PyTorch. Whether you're new to CUDA programming or a seasoned developer, you'll find examples and test cases to deepen your understanding of CUDA kernel generation.

## 🚀 Introduction

CUDA (Compute Unified Device Architecture) is the backbone of modern GPU programming, enabling highly efficient computation for deep learning, scientific simulations, and more. Writing efficient CUDA kernels, however, is often considered complex and time-consuming.

This project provides:
- Easy-to-follow examples of custom CUDA kernel generation.
- Test cases for advanced CUDA-powered operations, like attention mechanisms and relative position embeddings.
- Tools for benchmarking and optimizing PyTorch code with CUDA.

## 🌟 Features

- **Custom CUDA Kernels**: Examples of transforming PyTorch operations into CUDA for faster performance.
- **Advanced Attention Mechanisms**: Relative position embeddings, sliding window attention, PrefixLM, ALiBi, and more.
- **Customizable Block Masks**: Define unique attention patterns with CUDA.
- **Ease of Use**: All examples come with ready-to-use PyTorch models and initialization inputs.

---

## 📘 Getting Started

Follow these instructions to get started with the repository.

### Prerequisites
- Python 3.8+
- PyTorch with CUDA support
- A CUDA-enabled GPU


### Test Cases

#### Extremely basic example

```python

import torch
import torch.nn as nn
import time

#Define a simple PyTorch module
class SequentialOperations(nn.Module):
    def __init__(self):
        super(SequentialOperations, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        # Sequential operations
        x = x.cos()
        x = x.square()
        x = x.sin()
        x = self.relu(x)
        return x

```




#### Relative Position Embeddings

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Placeholder for the custom CUDA function
def relative_attention(query, key, value):
    """
    Placeholder function for attention with relative position encoding.
    This function will be replaced by a CUDA kernel.

    Args:
        query (torch.Tensor): Query tensor of shape (B, H, S, D).
        key (torch.Tensor): Key tensor of shape (B, H, S, D).
        value (torch.Tensor): Value tensor of shape (B, H, S, D).

    Returns:
        torch.Tensor: Output tensor of shape (B, H, S, D).
    """

    B, H, S, D = query.shape
    scores = torch.einsum("bhqd,bhkd->bhqk", query, key)  # Compute QK^T
    for q_idx in range(S):
        for kv_idx in range(S):
            scores[:, :, q_idx, kv_idx] += q_idx - kv_idx  # Apply relative position bias
    attention_weights = F.softmax(scores, dim=-1)

    # Compute weighted sum
    return torch.einsum("bhqk,bhvd->bhqd", attention_weights, value) 


class Model(nn.Module):
    """
    Model that performs scaled dot-product attention with relative position encoding.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, query, key, value):
        """
        Compute attention with relative position encoding.

        Args:
            query (torch.Tensor): Query tensor of shape (B, H, S, D).
            key (torch.Tensor): Key tensor of shape (B, H, S, D).
            value (torch.Tensor): Value tensor of shape (B, H, S, D).

        Returns:
            torch.Tensor: Output tensor of shape (B, H, S, D).
        """
        return relative_attention(query, key, value)


#Define the batch size, number of heads, sequence length, and embedding dimension
B = 8  # Batch size
H = 16  # Number of attention heads
S = 2048  # Sequence length
D = 128  # Embedding dimension per head

def get_inputs():
    """
    Generate random input tensors for query, key, and value.
    """
    query = torch.randn(B, H, S, D).cuda()
    key = torch.randn(B, H, S, D).cuda()
    value = torch.randn(B, H, S, D).cuda()
    return [query, key, value]

def get_init_inputs():
    """
    No special initialization inputs needed for this model.
    """
    return []
```




#### Custom Sliding Window Attention

```python

import torch
import torch.nn as nn
import torch.nn.functional as F

SLIDING_WINDOW = 2048

#Placeholder for the custom CUDA function
def sliding_window_attention(query, key, value):
    """
    Placeholder function for sliding window causal attention.
    This function will be replaced by a CUDA kernel.

    Args:
        query (torch.Tensor): Query tensor of shape (B, H, S, D).
        key (torch.Tensor): Key tensor of shape (B, H, S, D).
        value (torch.Tensor): Value tensor of shape (B, H, S, D).

    Returns:
        torch.Tensor: Output tensor of shape (B, H, S, D).
    """
    B, H, S, D = query.shape
    scores = torch.einsum("bhqd,bhkd->bhqk", query, key)  # Compute QK^T
    for q_idx in range(S):
        for kv_idx in range(S):
            causal_mask = q_idx >= kv_idx
            window_mask = q_idx - kv_idx <= SLIDING_WINDOW
            scores[:, :, q_idx, kv_idx] *= (causal_mask & window_mask)
    attention_weights = F.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bhvd->bhqd", attention_weights, value)  # Compute weighted sum


class Model(nn.Module):
    """
    Model that performs sliding window causal attention.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, query, key, value):
        """
        Compute sliding window causal attention.

        Args:
            query (torch.Tensor): Query tensor of shape (B, H, S, D).
            key (torch.Tensor): Key tensor of shape (B, H, S, D).
            value (torch.Tensor): Value tensor of shape (B, H, S, D).

        Returns:
            torch.Tensor: Output tensor of shape (B, H, S, D).
        """
        return sliding_window_attention(query, key, value)


#Define the batch size, number of heads, sequence length, and embedding dimension
B = 8  # Batch size
H = 16  # Number of attention heads
S = 2048  # Sequence length
D = 128  # Embedding dimension per head

def get_inputs():
    """
    Generate random input tensors for query, key, and value.
    """
    query = torch.randn(B, H, S, D).cuda()
    key = torch.randn(B, H, S, D).cuda()
    value = torch.randn(B, H, S, D).cuda()
    return [query, key, value]

def get_init_inputs():
    """
    No special initialization inputs needed for this model.
    """
    return []

```



#### PrefixLM Attention 

```python

import torch
import torch.nn as nn
import torch.nn.functional as F

#Placeholder for the custom CUDA function
def prefix_attention(query, key, value, prefix_length):
    """
    Placeholder function for PrefixLM attention with dynamic prefix-based and causal masking.
    This function will be replaced by a CUDA kernel.

    Args:
        query (torch.Tensor): Query tensor of shape (B, H, S, D).
        key (torch.Tensor): Key tensor of shape (B, H, S, D).
        value (torch.Tensor): Value tensor of shape (B, H, S, D).
        prefix_length (torch.Tensor): Tensor of shape (B,) indicating the prefix length for each sequence.

    Returns:
        torch.Tensor: Output tensor of shape (B, H, S, D).
    """
    B, H, S, D = query.shape
    scores = torch.einsum("bhqd,bhkd->bhqk", query, key)  # Compute QK^T

    for b in range(B):
        for q_idx in range(S):
            for kv_idx in range(S):
                causal_mask = q_idx >= kv_idx
                prefix_mask = kv_idx < prefix_length[b]
                scores[b, :, q_idx, kv_idx] *= (causal_mask or prefix_mask)

    attention_weights = F.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bhvd->bhqd", attention_weights, value)  # Compute weighted sum


class Model(nn.Module):
    """
    Model that performs PrefixLM attention with dynamic prefix-based and causal masking.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, query, key, value, prefix_length):
        """
        Compute PrefixLM attention.

        Args:
            query (torch.Tensor): Query tensor of shape (B, H, S, D).
            key (torch.Tensor): Key tensor of shape (B, H, S, D).
            value (torch.Tensor): Value tensor of shape (B, H, S, D).
            prefix_length (torch.Tensor): Tensor of shape (B,) indicating prefix length.

        Returns:
            torch.Tensor: Output tensor of shape (B, H, S, D).
        """
        return prefix_attention(query, key, value, prefix_length)


#Define the batch size, number of heads, sequence length, and embedding dimension

B = 8  # Batch size
H = 16  # Number of attention heads
S = 2048  # Sequence length
D = 128  # Embedding dimension per head

def get_inputs():
    """
    Generate random input tensors for query, key, value, and prefix length.
    """
    query = torch.randn(B, H, S, D).cuda()
    key = torch.randn(B, H, S, D).cuda()
    value = torch.randn(B, H, S, D).cuda()
    prefix_length = torch.randint(1, S, (B,), device="cuda")  # Random prefix lengths per batch
    return [query, key, value, prefix_length]

def get_init_inputs():
    """
    No special initialization inputs needed for this model.
    """
    return []

```



#### ALiBi Attention

```python

import torch
import torch.nn as nn
import torch.nn.functional as F

#Placeholder for the custom CUDA function
def alibi_attention(query, key, value, alibi_bias):
    """
    Placeholder function for ALiBi attention.
    This function will be replaced by a CUDA kernel.

    Args:
        query (torch.Tensor): Query tensor of shape (B, H, S, D).
        key (torch.Tensor): Key tensor of shape (B, H, S, D).
        value (torch.Tensor): Value tensor of shape (B, H, S, D).
        alibi_bias (torch.Tensor): ALiBi bias tensor of shape (H, S).

    Returns:
        torch.Tensor: Output tensor of shape (B, H, S, D).
    """
    B, H, S, D = query.shape
    scores = torch.einsum("bhqd,bhkd->bhqk", query, key)  # Compute QK^T
    scores += alibi_bias.unsqueeze(0).unsqueeze(2)  # Apply ALiBi bias
    attention_weights = F.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bhvd->bhqd", attention_weights, value)  # Compute weighted sum


class Model(nn.Module):
    """
    Model that performs ALiBi attention with precomputed linear biases.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, query, key, value, alibi_bias):
        """
        Compute ALiBi attention.

        Args:
            query (torch.Tensor): Query tensor of shape (B, H, S, D).
            key (torch.Tensor): Key tensor of shape (B, H, S, D).
            value (torch.Tensor): Value tensor of shape (B, H, S, D).
            alibi_bias (torch.Tensor): ALiBi bias tensor of shape (H, S).

        Returns:
            torch.Tensor: Output tensor of shape (B, H, S, D).
        """
        return alibi_attention(query, key, value, alibi_bias)


#Define the batch size, number of heads, sequence length, and embedding dimension
B = 8  # Batch size
H = 16  # Number of attention heads
S = 2048  # Sequence length
D = 128  # Embedding dimension per head

def get_inputs():
    """
    Generate random input tensors for query, key, value, and ALiBi bias.
    """
    query = torch.randn(B, H, S, D).cuda()
    key = torch.randn(B, H, S, D).cuda()
    value = torch.randn(B, H, S, D).cuda()
    # Example bias generation
    alibi_bias = torch.arange(S, device="cuda").unsqueeze(0).expand(H, S) * -0.1 
    return [query, key, value, alibi_bias]

def get_init_inputs():
    """
    No special initialization inputs needed for this model.
    """
    return []

```

#### Custom Attention Mask of Your choosing

```python

import torch
import torch.nn as nn
import torch.nn.functional as F

#Placeholder for the custom CUDA function
def custom_block_attention(query, key, value):
    """
    Placeholder function for block attention with a custom mask.
    This function will be replaced by a CUDA kernel.

    Args:
        query (torch.Tensor): Query tensor of shape (B, H, S, D).
        key (torch.Tensor): Key tensor of shape (B, H, S, D).
        value (torch.Tensor): Value tensor of shape (B, H, S, D).

    Returns:
        torch.Tensor: Output tensor of shape (B, H, S, D).
    """
    B, H, S, D = query.shape
    scores = torch.einsum("bhqd,bhkd->bhqk", query, key)  # Compute QK^T

    # Custom block mask: (i - j) % 4 == 0 below the diagonal
    mask = torch.zeros((S, S), device=query.device, dtype=torch.bool)
    for i in range(S):
        for j in range(i):  # Below the diagonal
            if (i - j) % 4 == 0:
                mask[i, j] = 1
    scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))  # Apply mask

    attention_weights = F.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bhvd->bhqd", attention_weights, value)  # Compute weighted sum


class Model(nn.Module):
    """
    Model that performs scaled dot-product attention with a custom block mask.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, query, key, value):
        """
        Compute block attention with a custom mask.

        Args:
            query (torch.Tensor): Query tensor of shape (B, H, S, D).
            key (torch.Tensor): Key tensor of shape (B, H, S, D).
            value (torch.Tensor): Value tensor of shape (B, H, S, D).

        Returns:
            torch.Tensor: Output tensor of shape (B, H, S, D).
        """
        return custom_block_attention(query, key, value)


#Define the batch size, number of heads, sequence length, and embedding dimension
B = 4  # Batch size
H = 8  # Number of attention heads
S = 64  # Sequence length
D = 32  # Embedding dimension per head

def get_inputs():
    """
    Generate random input tensors for query, key, and value.
    """
    query = torch.randn(B, H, S, D).cuda()
    key = torch.randn(B, H, S, D).cuda()
    value = torch.randn(B, H, S, D).cuda()
    return [query, key, value]

def get_init_inputs():
    """
    No special initialization inputs needed for this model.
    """
    return []

```

### LeNet


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_classes):
        """
        LeNet-5 architecture implementation in PyTorch.

        :param num_classes: The number of output classes.
        """
        super(Model, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
    
    def forward(self, x):
        """
        Forward pass of the LeNet-5 model.

        :param x: The input tensor, shape (batch_size, 1, 32, 32)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        # First convolutional layer with ReLU activation and max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Second convolutional layer with ReLU activation and max pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 16*5*5)
        
        # First fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))
        
        # Second fully connected layer with ReLU activation
        x = F.relu(self.fc2(x))
        
        # Final fully connected layer
        x = self.fc3(x)
        
        return x

# Test code for the LeNet-5 model
batch_size = 1
num_classes = 10

def get_inputs():
    return [torch.randn(batch_size, 1, 32, 32)]

def get_init_inputs():
    return [num_classes]
```
