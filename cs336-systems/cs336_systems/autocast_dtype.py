import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, in_features: int, hidden_dim:int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim, bias=False)
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_features, bias=False)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        print("dtype of model parameters fc1: ", self.fc1.weight.dtype)
        print("dtype of model parameters fc2: ", self.fc2.weight.dtype)
        print("dtype of model parameters ln: ", self.ln.weight.dtype)

        x = self.fc1(x)
        print("dtype of fc1 output: ", x.dtype)
        x = self.ln(x)
        print("dtype of ln output: ", x.dtype)
        x = self.relu(x)
        x = self.fc2(x)
        print("dtype of model's predicted logits: ", x.dtype)
        return x

device='cuda'
dtype=torch.bfloat16
train_context = torch.amp.autocast(device_type=device, dtype=dtype)
batch_size = 16
in_features = 32*32
hidden_dim = in_features*4
out_features = 10
steps = 1

model = ToyModel(in_features, hidden_dim, out_features)
model = model.to(device)
x = torch.randn(batch_size, in_features, device=device)

for _ in range(steps):
    with train_context:
        logits = model(x)
        loss = (logits**2).sum()
        print("dtype of loss: ", loss.dtype)
        loss.backward()
        print("dtype of model parameters fc1.grad: ", model.fc1.weight.grad.dtype)
        print("dtype of model parameters fc2.grad: ", model.fc2.weight.grad.dtype)
        print("dtype of model parameters ln.grad: ", model.ln.weight.grad.dtype)
        model.zero_grad()

