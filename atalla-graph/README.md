# Kernel Support Overview

## Kernels declared in `kernels/kernels.h`
All prototypes listed below have stub implementations in `kernels/kernels.c`:
- `matmul_kernel`, `add_kernel`, `mul_kernel`, `mul_scalar_kernel`
- `conv_kernel`
- `relu_kernel`, `softmax_kernel`
- `batchnorm_kernel`, `layernorm_kernel`
- `gelu_kernel`
- `avgpool_kernel`, `maxpool_kernel`
- `upsample_kernel`

## Ops currently lowered in `scripts/generate_schedule.py`
AlexNet-only lowering currently emits just these kernels:
- `torch.matmul`, `torch.ops.aten.matmul.default`
- `operator.add` / `torch.add` (with tensor×scalar mul folded into alpha)
- Tensor×tensor or tensor×scalar `mul`
- `F.conv2d`
- `F.relu` / `nn.ReLU`
- `F.avg_pool2d`, `nn.AdaptiveAvgPool2d`, `tensor.mean(...)`
- `nn.MaxPool2d`
- `F.softmax` / `nn.Softmax`

Other PyTorch ops (BatchNorm, LayerNorm, GELU, Upsample, etc.) still have kernel prototypes but are not emitted by `generate_schedule.py`.
Unsupported ops will cause an error. 
