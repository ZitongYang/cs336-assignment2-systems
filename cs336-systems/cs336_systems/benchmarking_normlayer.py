import torch
import time
from typing import Callable

from cs336_basics.model import RMSNorm
from cs336_systems.rmsnorm import RMSNormAutogradFuncTriton

# Constants
NUM_ROWS = 50000
HIDDEN_DIMS = [1024, 2048, 4096, 8192]
NUM_PASSES = 1000

# Time a single normalization layer's forward pass
def time_forward_pass(norm_layer: Callable,
                      input_tensor: torch.Tensor) -> float:
    start_time = time.time()
    for _ in range(NUM_PASSES):
        _ = norm_layer(input_tensor)
        torch.cuda.synchronize()  # Ensure accurate timing
    end_time = time.time()
    return (end_time - start_time) * 1000 / NUM_PASSES  # Return average time in milliseconds

# Main benchmarking function
def benchmark_normalization_layers():
    results = []
    for dim in HIDDEN_DIMS:
        # Random input tensors
        x = torch.randn(NUM_ROWS, dim, device=device)
        
        # Create LayerNorm and RMSNorm layers
        layer_norm = torch.nn.LayerNorm(dim).to(device)
        rms_norm = RMSNorm(dim).to(device)  # Assuming RMSNorm implemented similarly
        # Create Triton RMSNorm layer
        rms_norm_weights = torch.randn(dim, device=device)
        rms_norm_triton = lambda x: RMSNormAutogradFuncTriton.apply(x, rms_norm_weights)

        # Warm-up
        for _ in range(10):
            _ = layer_norm(x)
            _ = rms_norm(x)
            _ = rms_norm_triton(x)

        
        # Time the forward passes
        layer_norm_time = time_forward_pass(layer_norm, x)
        rms_norm_time = time_forward_pass(rms_norm, x)
        rms_norm_triton_time = time_forward_pass(rms_norm_triton, x)
        

        results.append((dim, layer_norm_time, rms_norm_time, rms_norm_triton_time))

    return results

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Execute benchmarking
    benchmark_results = benchmark_normalization_layers()

    # Print results in a markdown table format
    print("| Hidden Dimension  |LayerNorm Time (ms) |RMSNorm Time (ms)   |RMSNorm Triton Time (ms) |")
    print("|-------------------|--------------------|--------------------|-------------------------|")
    for dim, ln_time, rms_time, rms_triton_time in benchmark_results:
        print(f"| {dim}              | {ln_time:.2f}               | {rms_time:.2f}               | {rms_triton_time:.2f}                    |")
