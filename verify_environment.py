#!/usr/bin/env python3
"""
Setup: Verify Environment
Checks the lab environment and downloads the SmolLM-135M model.
"""

import os
import sys
import time


def verify_environment():
    """Verify all lab prerequisites are met."""
    print("=" * 65)
    print("vLLM Explained Lab - Environment Verification")
    print("=" * 65)

    checks_passed = 0
    checks_total = 5

    # Check 1: Virtual environment
    print("\n[1/5] Checking Python virtual environment...")
    if os.path.exists(".venv"):
        print("  PASS - Virtual environment found at .venv")
        checks_passed += 1
    else:
        print("  FAIL - Virtual environment not found")
        print("  Fix: Run 'uv venv --python 3.12'")
        return False

    # Check 2: Required packages
    print("\n[2/5] Checking required packages...")
    try:
        import torch
        import transformers
        print(f"  PASS - torch {torch.__version__}")
        print(f"  PASS - transformers {transformers.__version__}")
        checks_passed += 1
    except ImportError as e:
        print(f"  FAIL - Missing package: {e}")
        print("  Fix: Run 'uv pip install torch transformers'")
        return False

    # Check 3: vLLM
    print("\n[3/5] Checking vLLM installation...")
    try:
        import vllm
        print(f"  PASS - vllm {vllm.__version__}")
        checks_passed += 1
    except ImportError as e:
        print(f"  FAIL - vLLM not installed: {e}")
        print("  Fix: Run 'uv pip install vllm'")
        return False

    # Check 4: Additional packages
    print("\n[4/5] Checking additional packages...")
    try:
        import gradio
        import aiohttp
        import requests
        print(f"  PASS - gradio {gradio.__version__}")
        print(f"  PASS - aiohttp {aiohttp.__version__}")
        print(f"  PASS - requests {requests.__version__}")
        checks_passed += 1
    except ImportError as e:
        print(f"  FAIL - Missing package: {e}")
        print("  Fix: Run 'uv pip install gradio aiohttp requests'")
        return False

    # Check 5: Download SmolLM-135M model
    print("\n[5/5] Downloading SmolLM-135M model...")
    print("  This may take a minute on first run...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "HuggingFaceTB/SmolLM-135M"
        start_time = time.time()

        print(f"  Downloading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"  Downloading model for {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(model_name)

        elapsed = time.time() - start_time
        param_count = sum(p.numel() for p in model.parameters()) / 1e6

        print(f"  PASS - Model downloaded in {elapsed:.1f}s")
        print(f"  Model size: {param_count:.0f}M parameters")

        # Quick test generation
        print("\n  Running quick test generation...")
        inputs = tokenizer("Hello, world!", return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=10)
        test_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Test output: {test_output[:80]}...")
        print("  PASS - Model generates text successfully")

        # Clean up memory
        del model
        del tokenizer

        checks_passed += 1
    except Exception as e:
        print(f"  FAIL - Model download failed: {e}")
        return False

    # Summary
    print("\n" + "=" * 65)
    print(f"ENVIRONMENT CHECK: {checks_passed}/{checks_total} passed")
    print("=" * 65)

    if checks_passed == checks_total:
        print("\nAll checks passed! Your environment is ready.")
        print("\nLab Scenario:")
        print("  You are an ML engineer at InferenceIO.")
        print("  Mission: Use vLLM to serve SmolLM to concurrent users.")
        print("\nNext step: Run Task 1")
        print("  python task_1_hf_baseline.py")

        # Create marker
        os.makedirs("./markers", exist_ok=True)
        with open("./markers/environment_verified.txt", "w") as f:
            f.write("ENVIRONMENT_VERIFIED\n")

        print("\nEnvironment verification complete!")
        return True
    else:
        print(f"\n{checks_total - checks_passed} check(s) failed. Fix the issues above and retry.")
        return False


if __name__ == "__main__":
    success = verify_environment()
    sys.exit(0 if success else 1)
