#!/usr/bin/env python3
"""
Task 5: Serving with vLLM - OpenAI-Compatible API Server
Launches the vLLM OpenAI-compatible API server in the background,
and verifies it by sending a test chat completion request using the OpenAI client.
"""

import os
import sys
import time
import subprocess
import requests

def is_server_healthy():
    try:
        resp = requests.get("http://localhost:8000/health", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False

def main():
    print("=" * 65)
    print("Task 5: Serving with vLLM - OpenAI-Compatible API Server")
    print("=" * 65)

    # 1. Check if server is already running and healthy
    if is_server_healthy():
        print("vLLM API server is already running and healthy on port 8000.")
    else:
        print("Starting vLLM OpenAI-compatible API server in the background...")
        
        # Configure CPU-only execution environment
        env = os.environ.copy()
        env["VLLM_TARGET_DEVICE"] = "cpu"
        env["VLLM_CPU_KVCACHE_SPACE"] = "1"
        env["TORCHDYNAMO_DISABLE"] = "1"
        env["USE_LIBUV"] = "0"
        env["CUDA_VISIBLE_DEVICES"] = ""
        
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", "HuggingFaceTB/SmolLM-135M",
            "--port", "8000",
            "--max-model-len", "128",
            "--enforce-eager"
        ]
        
        os.makedirs("./markers", exist_ok=True)
        cmd_str = f'"{sys.executable}" -m vllm.entrypoints.openai.api_server --model HuggingFaceTB/SmolLM-135M --port 8000 --max-model-len 128 --enforce-eager > ./markers/vllm_server.log 2>&1'
        
        # Spawn the process in a detached state or a new process group so it persists
        # On Windows, we can use creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        # or just let it run. But using Popen without blocking is key.
        if sys.platform == "win32":
            # DETACHED_PROCESS = 0x00000008
            process = subprocess.Popen(
                cmd_str,
                env=env,
                creationflags=0x00000008,
                shell=True
            )
        else:
            process = subprocess.Popen(
                cmd_str,
                env=env,
                shell=True,
                preexec_fn=os.setpgrp
            )
            
        print("Waiting for server to initialize (polling health endpoint)...")
        start_time = time.time()
        timeout = 180  # 3 minutes maximum timeout on CPU
        healthy = False
        
        while time.time() - start_time < timeout:
            if is_server_healthy():
                healthy = True
                break
            time.sleep(3)
            elapsed = int(time.time() - start_time)
            print(f"  Waiting... {elapsed}s elapsed")
            
        if not healthy:
            print("ERROR: Server failed to start or become healthy within timeout.")
            sys.exit(1)
            
        print(f"vLLM API server is up and healthy! (took {int(time.time() - start_time)} seconds)")

    # 2. Configure OpenAI client
    print("\nConfiguring OpenAI client...")
    # TODO 1: Configure OpenAI client
    # Hint: Use the openai library to initialize a client pointing to localhost:8000/v1
    from openai import OpenAI
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-ignored-by-vllm"
    )

    # 3. Send test chat completion request
    print("Sending test request to server...")
    # TODO 2: Send chat completion request
    # Hint: Call client.chat.completions.create with model and messages
    prompt = "Explain what a large language model is in simple terms."
    start = time.time()
    
    response = client.completions.create(
        model="HuggingFaceTB/SmolLM-135M",
        prompt=f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        max_tokens=50,
        temperature=0.7
    )
    
    latency = time.time() - start
    generated_text = response.choices[0].text
    completion_tokens = response.usage.completion_tokens
    tps = completion_tokens / latency if latency > 0 else 0

    print("\n--- API RESPONSE ---")
    print(f"Generated text: {generated_text}")
    print(f"Completion tokens: {completion_tokens}")
    print(f"Latency: {latency:.2f} seconds")
    print(f"Throughput: {tps:.1f} tok/s")
    print("--------------------")

    # 4. Save marker file
    os.makedirs("./markers", exist_ok=True)
    with open("./markers/task5_complete.txt", "w") as f:
        f.write("TASK_5_COMPLETE\n")

    print("\nTask 5 Complete!")
    print("The vLLM server is running in the background on port 8000.")
    print("Next: python task_6_multi_user_load.py")

if __name__ == "__main__":
    main()
