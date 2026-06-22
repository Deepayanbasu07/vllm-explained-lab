#!/usr/bin/env python3
"""
Task 7: Parameter Tuning & Benchmarking
Benchmark different server configurations to find the optimal settings.
"""

import os
import sys
import time
import json
import asyncio
import subprocess
import requests
import aiohttp

URL = "http://localhost:8000/v1/completions"
MODEL = "HuggingFaceTB/SmolLM-135M"
PROMPT = "Explain the importance of cache memory in computers."
MAX_TOKENS = 30

def kill_server():
    """Kill any process listening on port 8000."""
    if sys.platform == "win32":
        try:
            output = subprocess.check_output("netstat -ano", shell=True, text=True)
            for line in output.splitlines():
                if ":8000" in line and "LISTENING" in line:
                    parts = line.strip().split()
                    pid = parts[-1]
                    subprocess.run(f"taskkill /F /PID {pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"  Killed process on port 8000 (PID {pid})")
                    time.sleep(2)
        except Exception as e:
            pass
    else:
        try:
            subprocess.run("fuser -k 8000/tcp", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2)
        except Exception:
            pass

def is_server_healthy():
    try:
        resp = requests.get("http://localhost:8000/health", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False

def start_server(max_model_len, max_num_seqs):
    """Launch server with specific settings."""
    env = os.environ.copy()
    env["VLLM_TARGET_DEVICE"] = "cpu"
    env["VLLM_CPU_KVCACHE_SPACE"] = "1"
    env["TORCHDYNAMO_DISABLE"] = "1"
    env["USE_LIBUV"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = ""
    
    cmd_str = f'"{sys.executable}" -m vllm.entrypoints.openai.api_server --model {MODEL} --port 8000 --max-model-len {max_model_len} --max-num-seqs {max_num_seqs} --enforce-eager > ./markers/vllm_server.log 2>&1'
    
    if sys.platform == "win32":
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
    
    # Wait for server to initialize
    start_time = time.time()
    timeout = 180
    healthy = False
    while time.time() - start_time < timeout:
        if is_server_healthy():
            healthy = True
            break
        time.sleep(2)
    
    if not healthy:
        raise RuntimeError("Server failed to initialize")
    
    return process

async def send_request(session):
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7
    }
    try:
        async with session.post(URL, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                tokens = data.get("usage", {}).get("completion_tokens", 0)
                return {"tokens": tokens, "success": True}
            return {"tokens": 0, "success": False}
    except Exception:
        return {"tokens": 0, "success": False}

async def run_benchmark(concurrency=10):
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session) for _ in range(concurrency)]
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time
        
        success_results = [r for r in results if r["success"]]
        total_tokens = sum(r["tokens"] for r in success_results)
        throughput = total_tokens / elapsed if elapsed > 0 else 0
        return throughput

def benchmark_config(config_name, max_model_len, max_num_seqs):
    print(f"\nBenchmarking config: '{config_name}'")
    print(f"  max_model_len={max_model_len}, max_num_seqs={max_num_seqs}")
    
    print("  Stopping existing server...")
    kill_server()
    
    print("  Starting server with new config...")
    start_server(max_model_len, max_num_seqs)
    
    print("  Running load benchmark...")
    loop = asyncio.get_event_loop()
    throughput = loop.run_until_complete(run_benchmark(concurrency=10))
    print(f"  Resulting Throughput: {throughput:.1f} tok/s")
    
    return {
        "config": config_name,
        "throughput": round(throughput, 2),
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs
    }

def main():
    print("=" * 65)
    print("Task 7: Parameter Tuning & Benchmarking")
    print("=" * 65)

    # TODO 1: Set max_model_len and TODO 2: Set max_num_seqs in configs
    configs = [
        {"name": "Default (len=128, seqs=16)", "max_model_len": 128, "max_num_seqs": 16},
        {"name": "Shorter Context (len=64)", "max_model_len": 64, "max_num_seqs": 16},
        {"name": "Limited Concurrency (seqs=4)", "max_model_len": 128, "max_num_seqs": 4}
    ]

    results = []
    for conf in configs:
        res = benchmark_config(conf["name"], conf["max_model_len"], conf["max_num_seqs"])
        results.append(res)

    # Save tuning results
    os.makedirs("./markers", exist_ok=True)
    with open("./markers/tuning_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Complete marker
    with open("./markers/task7_complete.txt", "w") as f:
        f.write("TASK_7_COMPLETE\n")

    print("\nTuning benchmarks complete! Results saved to ./markers/tuning_results.json")
    
    # Restart the default server for next tasks (Task 8 dashboard)
    print("\nRestarting default server for production use...")
    kill_server()
    start_server(128, 16)
    print("Default server is running on port 8000.")
    print("Next: python task_8_dashboard.py")

if __name__ == "__main__":
    main()
