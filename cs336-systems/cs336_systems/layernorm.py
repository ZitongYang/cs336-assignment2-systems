import torch
import time
from cs336_basics.model import RMSNorm

# Constants
NUM_ROWS = 50000
HIDDEN_DIMS = [1024, 2048, 4096, 8192]
NUM_PASSES = 1000

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Time a single normalization layer's forward pass
def time_forward_pass(norm_layer, input_tensor):
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

        # Warm-up
        for _ in range(10):
            _ = layer_norm(x)
            _ = rms_norm(x)
        
        # Time the forward passes
        layer_norm_time = time_forward_pass(layer_norm, x)
        rms_norm_time = time_forward_pass(rms_norm, x)

        results.append((dim, layer_norm_time, rms_norm_time))

    return results

# Execute benchmarking
benchmark_results = benchmark_normalization_layers()

# Print results in a markdown table format
print("| Hidden Dimension | LayerNorm Time (ms) | RMSNorm Time (ms) |")
print("|------------------|---------------------|-------------------|")
for dim, ln_time, rms_time in benchmark_results:
    print(f"| {dim}              | {ln_time:.2f}               | {rms_time:.2f}               |")
