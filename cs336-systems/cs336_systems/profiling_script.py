from dataclasses import dataclass, field, asdict
from typing import Optional
from transformers import HfArgumentParser
import torch
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from torch.profiler import profile, record_function, ProfilerActivity

from cs336_basics.nn_utils import cross_entropy, clip_gradient
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

# parsing the benchmarking configuration
@dataclass
class ProfilingConfig:
    # treatment variables for scaling
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    # optional arguments
    only_forward: Optional[bool] = field(default=False)
    profiling_iters: Optional[int] = field(default=5)
    warmup_iters: Optional[int] = field(default=1)
    wandb_run_name: Optional[str] = field(default='None')
    # fixed configs
    wandb_project: str = 'cs336-assignment2-systems'
    context_length: int = 128
    batch_size: int = 16
    vocab_size: int = 10000

    def __post_init__(self):
        self.wandb_logging = self.wandb_run_name != 'None'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


# parsing config
parser = HfArgumentParser(ProfilingConfig)
config = parser.parse_args_into_dataclasses()[0]
if config.wandb_logging:
    import wandb
    wandb.init(project=config.wandb_project, name=config.wandb_run_name)
logging.info(f'Benchmarking with config: {asdict(config)}')

# generate random dataset for bench marking
x = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length))
x = x.to(config.device)
y = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length))
y = y.to(config.device)

# initializing a rando model
model = BasicsTransformerLM(**asdict(config))
model = model.to(config.device)
# loading the optimizer
optimizer = AdamW(model.parameters())

def forward_pass():
    logits = model(x)
    loss = cross_entropy(logits, y)
    return loss

def backward_pass():
    optimizer.zero_grad()
    loss.backward()


for _ in range(config.warmup_iters):
    loss = forward_pass()
    backward_pass()
    clip_gradient(model.parameters(), 1.0)
    optimizer.step()

torch.cuda.synchronize()
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
    record_shapes=True,
    profile_memory=False,
    with_stack=True
) as prof:
    for _ in range(config.profiling_iters):
        with record_function("forward_pass"):
            loss = forward_pass()
            torch.cuda.synchronize()
            prof.step()
        if not config.only_forward:
            with record_function("backward_pass"):
                backward_pass()
                torch.cuda.synchronize()
                prof.step()
            with record_function("optimizer"):
                clip_gradient(model.parameters(), 1.0)
                optimizer.step()
                torch.cuda.synchronize()
                prof.step()


# benchmarking
prof.export_stacks(f"out/{config.wandb_run_name}_lm_profiler_stacks.txt", "self_cuda_time_total")
print(prof.key_averages().table(sort_by="cpu_time_total"))