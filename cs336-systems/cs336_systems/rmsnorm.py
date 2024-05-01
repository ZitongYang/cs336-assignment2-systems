import torch
import triton
import triton.language as tl
# import os
# os.environ['TRITON_INTERPRET']="1"

def _sum_all_but_last(x: torch.Tensor) -> torch.Tensor:
    return x.sum(dim=tuple(range(len(x.shape)-1)), keepdim=True)


@triton.jit
def _rmsnorm_fwd(x_ptr : tl.pointer_type,
                g_ptr : tl.pointer_type,
                y_ptr : tl.pointer_type,
                H : tl.uint32,
                eps: tl.float32,
                BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    x_row_start_ptr = x_ptr + row_idx * H
    y_row_start_ptr = y_ptr + row_idx * H

    # loading the vector needed for computation
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < H
    x_val = tl.load(x_row_start_ptr + offsets, mask=mask, other=0)
    g_val = tl.load(g_ptr + offsets, mask=mask, other=0)

    # computing the RMSNorm
    x_sq = x_val * x_val
    rms = tl.sqrt(tl.sum(x_sq) / H + eps)
    y_val = x_val * g_val / rms

    # storing the value
    tl.store(y_row_start_ptr + offsets, y_val, mask=mask)


@triton.jit
def _rmsnorm_bwd(x_ptr : tl.pointer_type,
                 g_ptr : tl.pointer_type, 
                 dout_ptr : tl.pointer_type,
                 dx_ptr : tl.pointer_type,
                 dg_ptr : tl.pointer_type,
                 H : tl.uint32,
                 eps: tl.float32,
                 BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    
    # loading corresponding dout, x row and g 
    xi_start_ptr = x_ptr + row_idx * H
    dout_start_ptr = dout_ptr + row_idx * H
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < H
    xi_val = tl.load(xi_start_ptr + offset, mask=mask, other=0)
    g_val = tl.load(g_ptr + offset, mask=mask, other=0)
    dout = tl.load(dout_start_ptr + offset, mask=mask, other=0)
    

    # compute some useful intermediate values
    denum = tl.sqrt(tl.sum(xi_val * xi_val) / H + eps)
    dg = dout * (xi_val/denum)
    dx = dout * (g_val/denum) - xi_val * tl.sum(dout * g_val * xi_val / (denum*denum*denum)) / H

    # store the results
    dx_start_ptr = dx_ptr + row_idx * H
    dg_start_ptr = dg_ptr + row_idx * H
    tl.store(dx_start_ptr + offset, dx, mask=mask)
    tl.store(dg_start_ptr + offset, dg, mask=mask)


class RMSNormAutogradFuncTriton(torch.autograd.Function):
    eps: float =1e-5

    @staticmethod
    def forward(ctx, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, g)
        
        # Set block size for the computation
        H = x.shape[-1]
        ctx.BLOCK_SIZE = triton.next_power_of_2(H)
        y = torch.empty(x.shape, device=x.device)

        # Check dimension and device consistency
        assert H == g.shape[0], "Dimension mismatch"
        assert x.is_cuda and g.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous() and g.is_contiguous(), "Our pointer arithmetic will assume contiguous x and g"

        n_row = int(y.numel() / H)
        # Launch our kernel with n_row instances in our 1D grid.
        _rmsnorm_fwd[(n_row,)](
            x, g, y, H, RMSNormAutogradFuncTriton.eps,
            num_warps=16, BLOCK_SIZE=ctx.BLOCK_SIZE
        )
        return y
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, g = ctx.saved_tensors
        H = x.shape[-1]
        BLOCK_SIZE = ctx.BLOCK_SIZE
        grad_x = torch.empty_like(x)
        grad_g = torch.empty_like(x)

        # Check dimension and device consistency
        assert H == g.shape[0], "Dimension mismatch"
        assert x.is_cuda and g.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous() and g.is_contiguous(), "Our pointer arithmetic will assume contiguous x and g"

        n_row = int(grad_output.numel() / H)
        _rmsnorm_bwd[(n_row,)](
            x, g, grad_output, grad_x, grad_g, H, RMSNormAutogradFuncTriton.eps,
            num_warps=16, BLOCK_SIZE=BLOCK_SIZE
        )
        return grad_x, _sum_all_but_last(grad_g)


class RMSNormTriton(torch.nn.Module):
    def __init__(self, H: int):
        super(RMSNormTriton, self).__init__()
        self.g = torch.nn.Parameter(torch.randn(H))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return RMSNormAutogradFuncTriton.apply(x, self.g)


class RMSNormAutogradFuncTorch(torch.autograd.Function):
    eps: float =1e-5

    def _jvp_g(dout: torch.Tensor, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        eps = RMSNormAutogradFuncTorch.eps
        denum = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        x_tilde = x/denum
        grad_g = _sum_all_but_last(dout * x_tilde)
        return grad_g

    def _jvp_x(dout: torch.Tensor, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        H = x.shape[-1]
        eps = RMSNormAutogradFuncTorch.eps
        denum = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        z = (1/H)*x*g/(denum**3)
        w = g/denum
        grad_x_1 = dout * w
        grad_x_2 = x * torch.sum(dout * z, dim=-1, keepdim=True)
        grad_x = grad_x_1 - grad_x_2
        return grad_x

    @staticmethod
    def forward(ctx, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, g)
        eps = RMSNormAutogradFuncTorch.eps
        denum = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        return x * g / denum
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, g = ctx.saved_tensors
        grad_x = RMSNormAutogradFuncTorch._jvp_x(grad_output, x, g)
        grad_g = RMSNormAutogradFuncTorch._jvp_g(grad_output, x, g)
        return grad_x, grad_g


        
if __name__ == '__main__':
    device = torch.device('cuda')

    # Test RMSNorm
    x = torch.randn((1, 4, 3), requires_grad=True, device=device)
    g = torch.ones(3, requires_grad=True, device=device)
    y_triton = RMSNormAutogradFuncTriton.apply(x, g)
    y_torch = RMSNormAutogradFuncTorch.apply(x, g)
    print('x matrices:\n', x)
    print('g vector:\n', g)
    print('y triton:\n', y_triton)
    print('y torch:\n', y_torch)