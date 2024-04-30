import torch
import triton
import triton.language as tl
# import os
# os.environ['TRITON_INTERPRET']="1"


@triton.jit
def _rmsnorm_fwd(x_ptr : tl.pointer_type,
                g_ptr : tl.pointer_type,
                y_ptr : tl.pointer_type,
                x_row_stride : tl.uint32,
                H : tl.uint32,
                eps: tl.float32,
                BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    x_row_start_ptr = x_ptr + row_idx * x_row_stride
    y_row_start_ptr = y_ptr + row_idx * x_row_stride

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
            x, g, y, x.shape[-1], H, RMSNormAutogradFuncTriton.eps,
            num_warps=16, BLOCK_SIZE=ctx.BLOCK_SIZE
        )
        return y


class RMSNormAutogradFuncTorch(torch.autograd.Function):
    eps: float =1e-5
    @staticmethod
    def forward(ctx, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, g)
        eps = RMSNormAutogradFuncTorch.eps
        denum = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        return x * g / denum
    
    # @staticmethod
    # def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
    #     x, g = ctx.saved_tensors
    #     eps = RMSNormAutogradFuncTorch.eps
    #     denum = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    #     grad_x = grad_output * g / denum
    #     grad_g = torch.sum(grad_output * x / denum, dim=0)
    #     return grad_x, grad_g

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


        
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