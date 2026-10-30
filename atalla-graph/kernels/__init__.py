"""AtallaC kernel generators.

Each module exposes a function that returns parameterised AtallaC source code
for a single kernel type. The generated C is compiled by ppci into Atalla
assembly and then encoded into .in files for the functional simulator.
"""

from .common import ADDR_TABLE, TILE, sdma_ctl_val, sdma_ctl_expr
from .gemm import gemm_c
from .relu import relu_c
from .softmax import softmax_c, softmax_c_batched
from .maxpool import maxpool_c
from .add import add_c
from .mul import mul_c
from .layernorm import layernorm_c
from .threshold_backward import threshold_backward_c
from .sum_dim import sum_dim_c
from .mse_loss import mse_loss_c
from .mse_loss_backward import mse_loss_backward_c
from .avgpool import avgpool_c
from .gelu import gelu_c
from .batchnorm import batchnorm_c
from .upsample import upsample_c
