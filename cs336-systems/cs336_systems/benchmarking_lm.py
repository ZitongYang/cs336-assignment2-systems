from dataclasses import dataclass, field, asdict
from typing import Optional, Callable
from transformers import HfArgumentParser
import torch
from contextlib import nullcontext
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import numpy as np
import time

from cs336_basics.nn_utils import cross_entropy, clip_gradient
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

# parsing the benchmarking configuration
@dataclass
class BenchMarkingConfig:
    # treatment variables for scaling
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    # optional arguments
    benchmarking_iters: Optional[int] = field(default=5)
    warmup_iters: Optional[int] = field(default=1)
    wandb_run_name: Optional[str] = field(default='None')
    mixed_precision: Optional[bool] = field(default=False)
    use_rms_norm: Optional[bool] = field(default=True)
    # fixed configs
    wandb_project: str = 'cs336-assignment2-systems'
    context_length: int = 128
    batch_size: int = 16
    vocab_size: int = 10000

    def __post_init__(self):
        self.wandb_logging = self.wandb_run_name != 'None'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


# parsing config
parser = HfArgumentParser(BenchMarkingConfig)
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
model = torch.compile(model)
# loading the optimizer
optimizer = AdamW(model.parameters())
# initialize the training context
if config.mixed_precision:
    train_context = torch.amp.autocast(device_type=config.device, dtype=torch.bfloat16)
else:
    train_context = nullcontext()

def forward_pass():
    torch.cuda.synchronize()
    logits = model(x)
    loss = cross_entropy(logits, y)
    torch.cuda.synchronize()
    return loss

def backward_pass():
    torch.cuda.synchronize()
    optimizer.zero_grad()
    loss.backward()
    torch.cuda.synchronize()

def timer(run: Callable):
    t1 = time.time()
    result = run()
    t2 = time.time()
    return t2-t1, result

iter_num = 0
# warm up
forward_times = np.zeros(config.benchmarking_iters)
backward_times = np.zeros(config.benchmarking_iters)
for _ in range(config.warmup_iters):
    with train_context:
        loss = forward_pass()
        backward_pass()
        clip_gradient(model.parameters(), 1.0)
        optimizer.step()

for i in range(config.benchmarking_iters):
    with train_context:
        forward_times[i], loss = timer(forward_pass)
        backward_times[i], _ = timer(backward_pass)
        clip_gradient(model.parameters(), 1.0)
        optimizer.step()


# benchmarking
print(f'Forward pass time: {np.mean(forward_times)}, std: {np.std(forward_times)}')
print(f'Backward pass time: {np.mean(backward_times)}, std: {np.std(backward_times)}')