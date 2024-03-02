import torch
import time

def benchmark_cpu_to_gpu_transfer(sizes_gb, trials=100):
    print("Testing CPU to GPU data transfer")
    print(f"{'Size (GB)':>10} {'Time (ms)':>10} {'Bandwidth (GB/s)':>20}")
    
    for size_gb in sizes_gb:
        size_bytes = size_gb * (1024 ** 3)  # Convert GB to bytes
        # Ensure the number of elements is an integer. float16 takes 2 bytes.
        num_elements = int(size_bytes // 2)  # Adjusted for float16, ensure this is an integer
        tensor = torch.randn(num_elements, dtype=torch.float16).pin_memory()
        tensor_gpu = torch.zeros_like(tensor, device="cuda")
        
        times = []
        for _ in range(trials):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            tensor_gpu.copy_(tensor, non_blocking=True)
            end.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded!
            elapsed_time_ms = start.elapsed_time(end)
            
            times.append(elapsed_time_ms)
        
        avg_time = sum(times) / trials
        bandwidth = (size_bytes / (1024 ** 3)) / (avg_time / 1000)  # GB/s
        print(f"{size_gb:>10} {avg_time:>10.2f} {bandwidth:>20.2f}")

sizes_gb = [0.01, 0.1, 1, 4, 8, 16, 32]  # Testing with 0.01GB (10MB), 0.1GB (100MB), and 1GB
benchmark_cpu_to_gpu_transfer(sizes_gb)
