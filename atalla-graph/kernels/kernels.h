#ifndef KERNELS_KERNELS_H
#define KERNELS_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#define MAX_TENSOR_RANK 8

typedef struct {
    uint32_t base_addr;
} TileDesc32;

typedef struct {
    uint8_t rank;
    uint32_t shape[MAX_TENSOR_RANK];
    uint32_t tiles_per_dim[MAX_TENSOR_RANK];
    size_t count;
    const TileDesc32 *tiles;
} GlobalTile;

int relu_kernel(const GlobalTile *input, GlobalTile *output, const void *vector_reg_base);
int softmax_kernel(const GlobalTile *input, GlobalTile *output);
int matmul_kernel(const GlobalTile *A, const GlobalTile *B, GlobalTile *C);
int add_kernel(const GlobalTile *lhs, const GlobalTile *rhs, GlobalTile *dst, float alpha);
int mul_kernel(const GlobalTile *lhs, const GlobalTile *rhs, GlobalTile *dst);
int mul_scalar_kernel(const GlobalTile *src, GlobalTile *dst, float scalar);
int conv_kernel(const GlobalTile *input, const GlobalTile *weight, const GlobalTile *bias, GlobalTile *output,
                int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int groups);
int batchnorm_kernel(const GlobalTile *input, const GlobalTile *weight, const GlobalTile *bias,
                     const GlobalTile *running_mean, const GlobalTile *running_var, GlobalTile *output, float eps);
int layernorm_kernel(const GlobalTile *input, const GlobalTile *weight, const GlobalTile *bias,
                     GlobalTile *output, int normalized_dim, float eps);
int gelu_kernel(const GlobalTile *input, GlobalTile *output, int approximate);
int avgpool_kernel(const GlobalTile *input, GlobalTile *output,
                   int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, int global_pool);
int maxpool_kernel(const GlobalTile *input, GlobalTile *output,
                   int kernel_h, int kernel_w, int stride_h, int stride_w,
                   int pad_h, int pad_w, int dilation_h, int dilation_w, int ceil_mode);
int upsample_kernel(const GlobalTile *input, GlobalTile *output, float scale_h, float scale_w, int mode, int align_corners);

#endif /* KERNELS_KERNELS_H */
