from dataclasses import dataclass, field, asdict
from typing import Optional
from transformers import HfArgumentParser
from contextlib import nullcontext
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
    mixed_precision: Optional[bool] = field(default=False)
    # fixed configs
    wandb_project: str = 'cs336-assignment2-systems'
    context_length: int = 128
    batch_size: int = 16
    vocab_size: int = 10000

    def __post_init__(self):
        self.wandb_logging = self.wandb_run_name != 'None'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.cuda.memory._record_memory_history(max_entries=1000000)
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
# initialize the training context
if config.mixed_precision:
    train_context = torch.amp.autocast(device_type=config.device, dtype=torch.bfloat16)
else:
    train_context = nullcontext()

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

torch.cuda.synchronize()
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA, ],
    schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=config.profiling_iters),
    experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with train_context:
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
        prof.export_memory_timeline("timeline.html", device=config.device)


# Save a pickle file to be loaded by PyTorch's online tool.
savename = f'{config.wandb_run_name}_forward_only.pickle' if config.only_forward else f'{config.wandb_run_name}_forward_backward.pickle'
torch.cuda.memory._dump_snapshot(f'out/{savename}')
# Stop recording history.
torch.cuda.memory._record_memory_history(enabled=None)