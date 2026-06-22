#!/usr/bin/env python3
"""
Task 3: KV Cache Problem - Contiguous Allocation Waste
Simulates memory allocation for 5 concurrent requests using standard
contiguous pre-allocation (worst-case scenario).
"""

import os

def main():
    print("=" * 65)
    print("Task 3: KV Cache Problem - Contiguous Allocation Waste")
    print("=" * 65)

    # 5 concurrent requests with different token lengths
    requests = [
        {"id": 1, "prompt_tokens": 45,  "description": "Short question"},
        {"id": 2, "prompt_tokens": 128, "description": "Medium paragraph"},
        {"id": 3, "prompt_tokens": 23,  "description": "Quick greeting"},
        {"id": 4, "prompt_tokens": 256, "description": "Long document"},
        {"id": 5, "prompt_tokens": 67,  "description": "Code snippet"},
    ]

    # TODO 1: Set the max sequence length (worst-case allocation size)
    # Hint: The model's maximum sequence length or typical pre-allocation limit is 512
    max_seq_len = 512  # TODO: Set to 512

    print(f"\nContiguous pre-allocation size: {max_seq_len} tokens per request")
    print("Simulating memory allocation for 5 requests:\n")

    total_allocated = 0
    total_used = 0

    for req in requests:
        actual = req["prompt_tokens"]
        allocated = max_seq_len
        
        total_allocated += allocated
        total_used += actual

        # TODO 2: Calculate waste percentage
        # Hint: (allocated - actual) / allocated * 100
        waste_percent = (allocated - actual) / allocated * 100  # TODO: Calculate waste percentage

        # Visual: show actual used slots vs wasted slots
        used_visual = "#" * int(actual / 10)
        waste_visual = "." * int((allocated - actual) / 10)
        print(f"  Request {req['id']} ({req['description']}):")
        print(f"    Used: {actual:>3} tokens, Allocated: {allocated:>3} tokens")
        print(f"    Memory Map: [{used_visual}{waste_visual}]")
        print(f"    Waste: {waste_percent:.1f}%")
        print("-" * 50)

    overall_waste = (total_allocated - total_used) / total_allocated * 100
    utilization = total_used / total_allocated * 100

    # --- SUMMARY ---
    print("\n--- SUMMARY (Contiguous Allocation) ---")
    print(f"Total Slots Allocated: {total_allocated}")
    print(f"Total Slots Used:      {total_used}")
    print(f"Overall Memory Waste:  {overall_waste:.1f}%")
    print(f"Memory Utilization:    {utilization:.1f}%")

    # --- KEY INSIGHT ---
    print("\n" + "=" * 65)
    print("KEY INSIGHT:")
    print("- Contiguous allocation forces us to allocate memory for the worst-case (512 tokens)")
    print("- This results in massive memory waste (~80% in this case)")
    print("- It limits model concurrency since we run out of memory serving empty slots")
    print("- Next: Let's see how PagedAttention solves this (Task 4)")
    print("=" * 65)

    # Save complete marker
    os.makedirs("./markers", exist_ok=True)
    with open("./markers/task3_complete.txt", "w") as f:
        f.write("TASK_3_COMPLETE\n")

    print("\nTask 3 Complete!")
    print("Next: python task_4_paged_attention.py")


if __name__ == "__main__":
    main()
