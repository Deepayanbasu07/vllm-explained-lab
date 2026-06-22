# vLLM Offline Inference & PagedAttention Lab: Windows CPU Optimization Guide

This repository contains a comprehensive, step-by-step hands-on laboratory designed to understand, simulate, benchmark, and deploy **vLLM (Virtual Large Language Model)** for high-throughput LLM serving. 

While vLLM is natively designed for Linux systems with high-end NVIDIA GPUs, **this project demonstrates a complete engineering port to Windows for local CPU-only execution**. By modifying core vLLM system internals and implementing custom PyTorch fallbacks, this lab operates entirely offline on a Windows workstation, serving the `SmolLM-135M` model.

---

## 🌟 Key Concepts Covered

### 1. The KV Cache Bottleneck
In autoregressive text generation, LLMs generate tokens sequentially. To generate the next token, the model needs the Key and Value (KV) activation tensors of all previous tokens. Storing these tensors in memory is known as the **KV Cache**.
* **The Problem:** HuggingFace's baseline implementation handles KV cache allocation contiguously. Memory is statically pre-allocated for each user based on the maximum sequence length (e.g., 512 tokens), regardless of the actual request length.
* **Resulting Waste:** If a user asks a short question generating 20 tokens, the remaining 492 tokens worth of KV cache memory are reserved but unused. This leads to **75% to 95% memory waste** (internal fragmentation) and severely limits the number of concurrent users a server can handle.

### 2. PagedAttention Mechanics
Inspired by OS virtual memory paging, **PagedAttention** partitions the KV cache of a request into fixed-size physical pages. 
* Each page holds keys and values for a small number of tokens (typically 16).
* The pages do not need to be contiguous in memory.
* A page table maps logical token sequences to physical blocks in memory.
* As the model generates new tokens, new pages are allocated dynamically, resulting in **~95% memory utilization** and allowing **4x to 5x higher serving concurrency**.

### 3. Continuous Batching
Instead of waiting for an entire batch of requests to finish generating before starting a new batch (static batching), vLLM implements **continuous batching** (iteration-level scheduling). New requests enter the batch immediately at the next iteration, and completed requests are released immediately.

### 4. Throughput Scaling
Under concurrent loads, vLLM schedules multiple sequences together. Since memory is managed dynamically, the cost of GPU/CPU memory transfers is amortized, causing the overall token generation throughput (tokens/second) to scale upwards as concurrency increases.

---

## 🛠️ Windows Porting: The Engineering Patches
Running vLLM on Windows CPU requires bypassing Linux-only system dependencies and GPU-compiled C++ extensions. The virtual environment in this project was successfully patched as follows:

1. **PyTorch CustomOp Fallback:** Edited `vllm/model_executor/custom_op.py` to catch `AttributeError` when importing compiled custom operations (`vllm._C`), routing execution to the PyTorch-native implementation.
2. **Pure PyTorch KV Cache Fallbacks:** Replaced C++ binding calls in `vllm/_custom_ops.py` for key operations (`reshape_and_cache`, `copy_blocks`, `swap_blocks`) with pure PyTorch tensors arithmetic.
3. **Pure PyTorch PagedAttention:** Patched `paged_attention_v1` inside `vllm/_custom_ops.py` to perform the scaled dot-product attention mathematically in PyTorch on CPU, removing compiled CUDA dependency.
4. **ZMQ Asynchronous Event Loop:** Patched `vllm/entrypoints/openai/api_server.py` and `vllm/entrypoints/cli/serve.py` to set the event loop policy to `WindowsSelectorEventLoopPolicy()`, enabling asynchronous ZeroMQ socket communications on Windows.
5. **Unix-only Signal Handlers & Sockets:** Removed Unix-only `SO_REUSEPORT` socket options and wrapped signal handlers (`signal.SIGINT`/`signal.SIGTERM`) in `vllm/entrypoints/launcher.py` with try-except guards.
6. **Prometheus Compatibility Bypass:** Commented out the `prometheus-fastapi-instrumentator` middleware initialization in `api_server.py` to prevent ASGI path parsing crashes on newer FastAPI versions.

---

## 🏃 Step-by-Step Lab Tasks

### 1. Environment Verification
Runs diagnostics on the virtual environment, verifies platform configuration, downloads the `HuggingFaceTB/SmolLM-135M` model, and checks that Python can load the weights.
```bash
python verify_environment.py
```

### 2. HuggingFace Baseline vs. vLLM Offline Inference
Compares single-request throughput. Task 1 loads the model via standard HuggingFace `AutoModelForCausalLM` and measures tokens/second. Task 2 initializes the local vLLM offline engine and runs the same prompt.
```bash
python task_1_hf_baseline.py
python task_2_vllm_inference.py
```

### 3. KV Cache Contiguous Allocation Simulation
Simulates serving 5 concurrent requests of varying lengths (e.g., 23 to 256 tokens) under a traditional system. It pre-allocates 512 tokens per request, visualizing the memory maps and showing **~80% overall memory waste**.
```bash
python task_3_kv_cache_problem.py
```

### 4. PagedAttention Memory Simulation
Runs the same 5 requests but uses virtual memory paging. It divides the requests into pages of 16 tokens and dynamically allocates only what is needed, demonstrating **~95% memory utilization** and a **4.8x increase in concurrency**.
```bash
python task_4_paged_attention.py
```

### 5. OpenAI-Compatible API Serving
Launches the vLLM OpenAI-compatible server in the background on port `8000` (CPU-optimized eager mode) and runs a test completions query using the official `openai` Python SDK.
```bash
python task_5_api_server.py
```

### 6. Multi-User Load Testing
Uses `aiohttp` to send concurrent asynchronous requests (concurrency levels: 1, 5, 10, and 20 users) to the running local server, measuring how overall throughput (tok/s) scales under load.
```bash
python task_6_multi_user_load.py
```

### 7. Parameter Tuning Benchmarks
Runs automated benchmarks across three server configurations to analyze performance trade-offs:
1. **Default:** `max_model_len=128`, `max_num_seqs=16`
2. **Shorter Context:** `max_model_len=64`
3. **Limited Concurrency:** `max_num_seqs=4`
```bash
python task_7_tuning.py
```

### 8. Capstone Gradio Dashboard
Starts an interactive web monitoring dashboard on port `7860`. The dashboard displays live latency/throughput stats, side-by-side speed comparison charts, concurrency scaling plots, parameter tuning results, and a summary table of the entire lab journey.
```bash
python task_8_dashboard.py
```

---

## 📊 Summary of Benchmarks & Results (Local CPU)

* **HuggingFace Baseline (Single Request):** ~32.4 tok/s
* **vLLM Baseline (Single Request):** ~28.5 tok/s
* **KV Cache Contiguous Waste:** **79.7%**
* **PagedAttention Utilization:** **95.4%**
* **Concurrency Throughput Scaling (vLLM API Server):**
  * **1 User:** 26.6 tok/s
  * **5 Users:** 78.6 tok/s
  * **10 Users:** 102.1 tok/s
  * **20 Users:** **118.4 tok/s** (Demonstrating continuous batching scaling)
* **Configuration Performance Tuning:**
  * *Default config (len=128):* 99.2 tok/s
  * *Shorter context config (len=64):* 101.0 tok/s
  * *Limited concurrency (seqs=4):* 63.9 tok/s

---

## 🚀 Setup & Execution Guide

### Prerequisites
* Windows 10 or 11
* Python 3.12 (Virtual Environment recommended)
* [uv](https://github.com/astral-sh/uv) (for ultra-fast package installation)

### Install Dependencies & Run Lab
```bash
# Clone the repository
git clone <your-repo-url>
cd VLLM

# Create virtual environment and install packages
uv venv
uv pip install -r requirements.txt
uv pip install matplotlib

# Run tasks sequentially
python verify_environment.py
python task_1_hf_baseline.py
python task_2_vllm_inference.py
python task_3_kv_cache_problem.py
python task_4_paged_attention.py
python task_5_api_server.py
python task_6_multi_user_load.py
python task_7_tuning.py
python task_8_dashboard.py
```

Once Task 8 is running, open **`http://localhost:7860`** in your browser to view the interactive dashboard.
