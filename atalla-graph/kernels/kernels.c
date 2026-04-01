#include "kernels.h"

int relu_kernel(const GlobalTile *input, GlobalTile *output, const void *vector_reg_base) {
    (void)input;
    (void)output;
    (void)vector_reg_base;
    return 0;
}

int softmax_kernel(const GlobalTile *input, GlobalTile *output) {
    (void)input;
    (void)output;
    return 0;
}

int matmul_kernel(const GlobalTile *A, const GlobalTile *B, GlobalTile *C) {
    (void)A;
    (void)B;
    (void)C;
    return 0;
}

int add_kernel(const GlobalTile *lhs, const GlobalTile *rhs, GlobalTile *dst, float alpha) {
    (void)lhs;
    (void)rhs;
    (void)dst;
    (void)alpha;
    return 0;
}

int mul_kernel(const GlobalTile *lhs, const GlobalTile *rhs, GlobalTile *dst) {
    (void)lhs;
    (void)rhs;
    (void)dst;
    return 0;
}

int mul_scalar_kernel(const GlobalTile *src, GlobalTile *dst, float scalar) {
    (void)src;
    (void)dst;
    (void)scalar;
    return 0;
}

int conv_kernel(const GlobalTile *input, const GlobalTile *weight, const GlobalTile *bias, GlobalTile *output,
                int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int groups) {
    (void)input;
    (void)weight;
    (void)bias;
    (void)output;
    (void)stride_h;
    (void)stride_w;
    (void)pad_h;
    (void)pad_w;
    (void)dilation_h;
    (void)dilation_w;
    (void)groups;
    return 0;
}

int batchnorm_kernel(const GlobalTile *input, const GlobalTile *weight, const GlobalTile *bias,
                     const GlobalTile *running_mean, const GlobalTile *running_var, GlobalTile *output, float eps) {
    (void)input;
    (void)weight;
    (void)bias;
    (void)running_mean;
    (void)running_var;
    (void)output;
    (void)eps;
    return 0;
}

int layernorm_kernel(const GlobalTile *input, const GlobalTile *weight, const GlobalTile *bias,
                     GlobalTile *output, int normalized_dim, float eps) {
    (void)input;
    (void)weight;
    (void)bias;
    (void)output;
    (void)normalized_dim;
    (void)eps;
    return 0;
}

int gelu_kernel(const GlobalTile *input, GlobalTile *output, int approximate) {
    (void)input;
    (void)output;
    (void)approximate;
    return 0;
}

int avgpool_kernel(const GlobalTile *input, GlobalTile *output,
                   int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, int global_pool) {
    (void)input;
    (void)output;
    (void)kernel_h;
    (void)kernel_w;
    (void)stride_h;
    (void)stride_w;
    (void)pad_h;
    (void)pad_w;
    (void)global_pool;
    return 0;
}

int maxpool_kernel(const GlobalTile *input, GlobalTile *output,
                   int kernel_h, int kernel_w, int stride_h, int stride_w,
                   int pad_h, int pad_w, int dilation_h, int dilation_w, int ceil_mode) {
    (void)input;
    (void)output;
    (void)kernel_h;
    (void)kernel_w;
    (void)stride_h;
    (void)stride_w;
    (void)pad_h;
    (void)pad_w;
    (void)dilation_h;
    (void)dilation_w;
    (void)ceil_mode;
    return 0;
}

int upsample_kernel(const GlobalTile *input, GlobalTile *output, float scale_h, float scale_w, int mode, int align_corners) {
    (void)input;
    (void)output;
    (void)scale_h;
    (void)scale_w;
    (void)mode;
    (void)align_corners;
    return 0;
}
