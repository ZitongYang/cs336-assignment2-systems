import torch
import time
from typing import Callable, Tuple

from cs336_basics.model import RMSNorm
from cs336_systems.rmsnorm import RMSNormTriton

# Constants
NUM_ROWS = 50000
HIDDEN_DIMS = [1024, 2048, 4096, 8192]
NUM_PASSES = 1000

# Time a single normalization layer's forward pass
def time_forward_backward(norm_layer: Callable,
                         input_tensor: torch.Tensor,
                         upstream_gradient: torch.Tensor) -> Tuple[float]:
    forward_time = 0.0
    backward_time = 0.0
    for _ in range(NUM_PASSES):
        t1 = time.time()
        output = norm_layer(input_tensor)
        torch.cuda.synchronize()  # Ensure accurate timing
        t2 = time.time()
        forward_time += t2 - t1
        output.backward(upstream_gradient)
        torch.cuda.synchronize()  # Ensure accurate timing
        t3 = time.time()
        backward_time += t3 - t2
    return forward_time * 1000 / NUM_PASSES, backward_time * 1000 / NUM_PASSES  # Return average time in milliseconds

# Main benchmarking function
def benchmark_normalization_layers():
    results = []
    for dim in HIDDEN_DIMS:
        # Random input tensors
        x = torch.randn(NUM_ROWS, dim, device=device, requires_grad=True)
        dout = torch.randn(NUM_ROWS, dim, device=device)
        
        # Create LayerNorm and RMSNorm layers
        layer_norm = torch.nn.LayerNorm(dim).to(device)
        rms_norm = RMSNorm(dim).to(device)  # Assuming RMSNorm implemented similarly
        # Create Triton RMSNorm layer
        rms_norm_triton = RMSNormTriton(dim).to(device)


        # Warm-up
        for _ in range(10):
            _ = layer_norm(x)
            _ = rms_norm(x)
            _ = rms_norm_triton(x)

        
        # Time the forward passes
        layer_norm_time = time_forward_backward(layer_norm, x, dout)
        rms_norm_time = time_forward_backward(rms_norm, x, dout)
        rms_norm_triton_time = time_forward_backward(rms_norm_triton, x, dout)
        

        results.append((dim, layer_norm_time, rms_norm_time, rms_norm_triton_time))

    return results

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Execute benchmarking
    benchmark_results = benchmark_normalization_layers()

    # # Print results in a markdown table format with only forward pass
    # print("| Hidden Dimension  |LayerNorm Time (ms) |RMSNorm Time (ms)   |RMSNorm Triton Time (ms) |")
    # print("|-------------------|--------------------|--------------------|-------------------------|")
    # for dim, ln_time, rms_time, rms_triton_time in benchmark_results:
    #     print(f"| {dim}              | {ln_time[0]:.2f}               | {rms_time[0]:.2f}               | {rms_triton_time[0]:.2f}                    |")

    # Print results in a markdown table format with both forward and backward pass  
    print("| Hidden Dimension  |LayerNorm Forward (ms) |RMSNorm Forward (ms) |RMSNorm Triton Forward (ms) |LayerNorm Backward (ms) |RMSNorm Backward (ms) |RMSNorm Triton Backward (ms) |")
    print("|-------------------|------------------------|----------------------|---------------------------|-----------------------|---------------------|----------------------------|")
    for dim, ln_time, rms_time, rms_triton_time in benchmark_results:
        print(f"| {dim}              | {ln_time[0]:.2f}                   | {rms_time[0]:.2f}                 | {rms_triton_time[0]:.2f}                      | {ln_time[1]:.2f}               | {rms_time[1]:.2f}             | {rms_triton_time[1]:.2f}                      |")
