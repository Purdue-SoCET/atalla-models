"""Named Linalg convolutions: validation and the canonical execution plan.

Every supported ``linalg.conv_*`` op is the same loop nest once its operands
are seen through byte strides:

    O[n, g, o0, o1, o2, f] += cast(I[n, g, o0*s0 + k0*d0, o1*s1 + k1*d1,
                                      o2*s2 + k2*d2, c]) * cast(K[g, f, k0, k1, k2, c])

(``_q`` ops subtract the input/filter zero points after the cast.) ``OPS`` maps
each named op's operand axes onto that canonical nest; missing axes get extent
one. The plan picks one execution path; the C engine in ``kernels/conv_engine.h``
follows it:

``array``   bf16 implicit GEMM on the systolic array. Activation rows and weight
            columns run along ``c``; the output tile stays in SP1 for every tap
            and channel tile. An operand whose reduction axis (input, filter) or
            filter axis (output) is not contiguous is first restaged on device
            into a channel-last workspace by array transposes (``Restage``).
``lanes``   bf16 channel-free ``conv_1d/2d/3d`` on the vector lanes: 32 outputs
            per vector, one ``mul.vs`` + ``add.vv`` per tap in Linalg order.
``scalar``  everything else (integer and quantized types, odd alignment,
            unsupported strides): scalar loops in Linalg's reduction order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from graph.layout import ELEMENT_BYTES, TensorView, _index_range, _integers, validate_view

# name -> (spatial rank, input axes, filter axes, output axes, reduction order, quantized)
# Axis tokens: N G C F, S0..S2 input spatial, K0..K2 filter taps, O0..O2 output spatial.
OPS = {
    'conv_1d': (1, 'S0', 'K0', 'O0', 'K0', False),
    'conv_2d': (2, 'S0 S1', 'K0 K1', 'O0 O1', 'K0 K1', False),
    'conv_3d': (3, 'S0 S1 S2', 'K0 K1 K2', 'O0 O1 O2', 'K0 K1 K2', False),
    'conv_1d_nwc_wcf': (1, 'N S0 C', 'K0 C F', 'N O0 F', 'K0 C', False),
    'conv_1d_ncw_fcw': (1, 'N C S0', 'F C K0', 'N F O0', 'C K0', False),
    'conv_2d_nhwc_hwcf': (2, 'N S0 S1 C', 'K0 K1 C F', 'N O0 O1 F', 'K0 K1 C', False),
    'conv_2d_nhwc_fhwc': (2, 'N S0 S1 C', 'F K0 K1 C', 'N O0 O1 F', 'K0 K1 C', False),
    'conv_2d_nhwc_hwcf_q': (2, 'N S0 S1 C', 'K0 K1 C F', 'N O0 O1 F', 'K0 K1 C', True),
    'conv_2d_nhwc_fhwc_q': (2, 'N S0 S1 C', 'F K0 K1 C', 'N O0 O1 F', 'K0 K1 C', True),
    'conv_2d_nchw_fchw': (2, 'N C S0 S1', 'F C K0 K1', 'N F O0 O1', 'C K0 K1', False),
    'conv_2d_nchw_fchw_q': (2, 'N C S0 S1', 'F C K0 K1', 'N F O0 O1', 'C K0 K1', True),
    'conv_2d_ngchw_fgchw': (2, 'N G C S0 S1', 'F G C K0 K1', 'N G F O0 O1', 'C K0 K1', False),
    'conv_2d_ngchw_gfchw': (2, 'N G C S0 S1', 'G F C K0 K1', 'N G F O0 O1', 'C K0 K1', False),
    'conv_2d_ngchw_gfchw_q': (2, 'N G C S0 S1', 'G F C K0 K1', 'N G F O0 O1', 'C K0 K1', True),
    'conv_2d_nhwgc_gfhwc': (2, 'N S0 S1 G C', 'G F K0 K1 C', 'N O0 O1 G F', 'K0 K1 C', False),
    'conv_2d_nhwgc_gfhwc_q': (2, 'N S0 S1 G C', 'G F K0 K1 C', 'N O0 O1 G F', 'K0 K1 C', True),
    'conv_3d_ndhwc_dhwcf': (3, 'N S0 S1 S2 C', 'K0 K1 K2 C F', 'N O0 O1 O2 F', 'K0 K1 K2 C', False),
    'conv_3d_ndhwc_dhwcf_q': (3, 'N S0 S1 S2 C', 'K0 K1 K2 C F', 'N O0 O1 O2 F', 'K0 K1 K2 C', True),
    'conv_3d_ncdhw_fcdhw': (3, 'N C S0 S1 S2', 'F C K0 K1 K2', 'N F O0 O1 O2', 'C K0 K1 K2', False),
}
PLAIN = ('conv_1d', 'conv_2d', 'conv_3d')

DTYPE_CODE = {'bf16': 0, 'i8': 1, 'i16': 2, 'i32': 3}
INTEGER_INPUTS = ('i8', 'i16', 'i32')
PATH_CODE = {'none': 0, 'scalar': 1, 'array': 2, 'lanes': 3}
MAX_PITCH = 1 << 20          # SDMA full_num_cols field, in elements
TILE = 32


@dataclass(frozen=True)
class ConvOp:
    name: str
    source: TensorView
    weights: TensorView
    destination: TensorView
    strides: tuple = ()
    dilations: tuple = ()
    input_zero_point: int | None = None
    filter_zero_point: int | None = None

    def __post_init__(self):
        for attr in ('strides', 'dilations'):
            object.__setattr__(self, attr, _integers(getattr(self, attr)))


@dataclass(frozen=True)
class Restage:
    """On-device transpose of a batch of 2D matrices (32x32 array tiles).

    Source element (b, r, c) lives at ``src + b.ss + r*src_row + 2c``; it is
    written to ``dst + b.ds + c*dst_row + 2r``. Addresses are buffer-relative
    (``*_buffer`` names an allocation or a workspace) and resolved by the emitter.
    """
    src_buffer: str
    src_offset: int
    dst_buffer: str
    dst_offset: int
    batch: tuple            # 4 extents
    src_batch: tuple        # 4 byte strides
    dst_batch: tuple        # 4 byte strides
    rows: int
    cols: int
    src_row: int
    dst_row: int


@dataclass(frozen=True)
class ConvPlan:
    name: str
    path: str
    dtypes: tuple           # input, filter, output
    zero_points: tuple      # input, filter (0 when not quantized)
    extents: dict           # N G F C O0 O1 O2 K0 K1 K2
    steps: tuple
    dilations: tuple
    # Canonical operand views used by the compute loop: (buffer, offset, strides).
    # Input strides: N G S0 S1 S2 C; filter: G F K0 K1 K2 C; output: N G O0 O1 O2 F.
    input: tuple
    filter: tuple
    output: tuple
    reduction: tuple        # Linalg order of C/K0/K1/K2 for the scalar path
    pre: tuple = ()         # Restage before compute
    post: tuple = ()        # Restage after compute
    workspaces: dict = field(default_factory=dict)   # name -> bytes
    elements: int = 0
    macs: int = 0
    fold: int = 1           # k2 folded into the reduction (array path): original K2


def _axes(spec):
    return spec.split()


def _canonical(view: TensorView, axes, order):
    """Byte strides of ``view`` for the canonical ``order`` (0 for absent axes)."""
    stride = dict(zip(axes, view.strides_bytes))
    return tuple(stride.get(a, 0) for a in order)


def lower_conv(op: ConvOp, buffers) -> ConvPlan:
    name = op.name.removeprefix('linalg.')
    if name not in OPS:
        raise NotImplementedError(f'unsupported convolution: {op.name}')
    rank, i_spec, k_spec, o_spec, red_spec, quantized = OPS[name]
    # Right-align the named spatial axes into the canonical three (axis 2 innermost).
    shift = 3 - rank
    rename = lambda spec: [a if a[0] not in 'SKO' else f'{a[0]}{int(a[1]) + shift}'
                           for a in spec.split()]
    i_axes, k_axes, o_axes, red = (rename(x) for x in (i_spec, k_spec, o_spec, red_spec))
    red = tuple(red)
    src, wts, dst = op.source, op.weights, op.destination
    validate_view(src, buffers)
    validate_view(wts, buffers)
    validate_view(dst, buffers, writable=True)
    if (len(src.shape), len(wts.shape), len(dst.shape)) != (len(i_axes), len(k_axes), len(o_axes)):
        raise ValueError(f'{name}: operand ranks do not match the named layout')

    # Element types: matching bf16, or signed integers accumulated in i32.
    dtypes = (src.dtype, wts.dtype, dst.dtype)
    if quantized:
        if src.dtype not in INTEGER_INPUTS or wts.dtype not in INTEGER_INPUTS or dst.dtype != 'i32':
            raise NotImplementedError(f'{name}: quantized convolution needs integer inputs and i32 output')
        if op.input_zero_point is None or op.filter_zero_point is None:
            raise ValueError(f'{name}: quantized convolution needs both zero points')
        zps = (int(op.input_zero_point), int(op.filter_zero_point))
        for z in zps:
            if not -(1 << 31) <= z < (1 << 31):
                raise OverflowError('zero point exceeds i32')
    else:
        if op.input_zero_point is not None or op.filter_zero_point is not None:
            raise ValueError(f'{name}: zero points apply only to quantized convolutions')
        zps = (0, 0)
        if dtypes != ('bf16',) * 3 and not (src.dtype in INTEGER_INPUTS and wts.dtype in INTEGER_INPUTS
                                             and dst.dtype == 'i32'):
            raise NotImplementedError(f'{name}: supports bf16 operands, or integer inputs with i32 output')

    # Extents per canonical axis; shared axes must agree across operands.
    ext = {a: 1 for a in ('N', 'G', 'F', 'C', 'O0', 'O1', 'O2', 'K0', 'K1', 'K2')}
    sext = {'S0': 1, 'S1': 1, 'S2': 1}
    seen = {}
    for view, axes in ((src, i_axes), (wts, k_axes), (dst, o_axes)):
        for a, size in zip(axes, view.shape):
            if a[0] == 'S':
                sext[a] = size
            elif seen.setdefault(a, size) != size:
                raise ValueError(f'{name}: extent of {a} disagrees between operands')
            else:
                ext[a] = size

    steps = op.strides or (1,) * rank
    dilations = op.dilations or (1,) * rank
    if name in PLAIN and (op.strides or op.dilations):
        raise ValueError(f'{name} has no strides or dilations')
    if len(steps) != rank or len(dilations) != rank or min(steps + dilations) <= 0:
        raise ValueError(f'{name}: strides and dilations must be positive, one per spatial axis')
    for v in steps + dilations:
        _index_range(v)
    steps = (1,) * shift + tuple(steps)
    dilations = (1,) * shift + tuple(dilations)

    elements = math.prod(dst.shape)
    reduction = ext['C'] * ext['K0'] * ext['K1'] * ext['K2']
    if elements and reduction:
        for d in range(3):
            need = (ext[f'O{d}'] - 1) * steps[d] + (ext[f'K{d}'] - 1) * dilations[d] + 1
            if need > sext[f'S{d}']:
                raise ValueError(f'{name}: output window exceeds input along spatial axis {d}')

    i_order = ('N', 'G', 'S0', 'S1', 'S2', 'C')
    k_order = ('G', 'F', 'K0', 'K1', 'K2', 'C')
    o_order = ('N', 'G', 'O0', 'O1', 'O2', 'F')
    i_str = _canonical(src, i_axes, i_order)
    k_str = _canonical(wts, k_axes, k_order)
    o_str = _canonical(dst, o_axes, o_order)
    plan = dict(name=name, dtypes=dtypes, zero_points=zps, extents=ext, steps=steps,
                dilations=dilations, reduction=red, elements=elements,
                macs=elements * reduction,
                input=(src.buffer, src.offset_bytes, i_str),
                filter=(wts.buffer, wts.offset_bytes, k_str),
                output=(dst.buffer, dst.offset_bytes, o_str))
    if not elements or not reduction:
        return ConvPlan(path='none', **plan)
    if dtypes != ('bf16',) * 3 or _aliased(op):
        return ConvPlan(path='scalar', **plan)
    if name in PLAIN:
        return ConvPlan(path='lanes' if _lanes_ok(plan) else 'scalar', **plan)
    return _array_plan(plan)


def _aliased(op):
    return op.destination.buffer in (op.source.buffer, op.weights.buffer)


def _even(*values):
    return all(v % 2 == 0 for v in values)


def _pitch_ok(stride_bytes):
    return stride_bytes > 0 and stride_bytes % 2 == 0 and stride_bytes // 2 < MAX_PITCH


def _lanes_ok(p):
    (_, ioff, i), (_, koff, k), (_, ooff, o) = p['input'], p['filter'], p['output']
    return i[4] == 2 and o[4] == 2 and _even(ioff, koff, ooff, *i, *k, *o)


def _array_plan(base):
    """Choose restaging so the compute loop sees c-contiguous input/filter and
    f-contiguous output; fall back to scalar when a view cannot be restaged."""
    p = dict(base)
    scalar = ConvPlan(path='scalar', **base)
    e = p['extents']
    N, G, F, C = e['N'], e['G'], e['F'], e['C']
    T = (e['K0'], e['K1'], e['K2'])
    O = (e['O0'], e['O1'], e['O2'])
    ibuf, ioff, i = p['input']
    kbuf, koff, k = p['filter']
    obuf, ooff, o = p['output']
    if not _even(ioff, koff, ooff, *i, *k, *o):
        return scalar
    pre, post, ws = [], [], {}

    # Input -> [N G S0 S1 S2 C], C contiguous; rows c, cols s2 per (n, g, s0, s1).
    if i[5] != 2:
        if i[4] != 2:
            return scalar
        S = [(O[d] - 1) * p['steps'][d] + (T[d] - 1) * p['dilations'][d] + 1 for d in range(3)]
        w = dict(c=2, s2=2 * C, s1=2 * C * S[2], s0=2 * C * S[2] * S[1], g=2 * C * math.prod(S))
        w['n'] = w['g'] * G
        ws['input'] = w['n'] * N
        pre.append(Restage(ibuf, ioff, 'input', 0, (N, G, S[0], S[1]), (i[0], i[1], i[2], i[3]),
                           (w['n'], w['g'], w['s0'], w['s1']), C, S[2], i[5], w['s2']))
        p['input'] = ('input', 0, (w['n'], w['g'], w['s0'], w['s1'], w['s2'], w['c']))

    # Filter -> [G F K0 K1 K2 C], C contiguous (one weight column per f); filter-major
    # so that (k2, c) is one contiguous run and can fold into the reduction.
    if k[5] != 2:
        w = dict(c=2, k2=2 * C, k1=2 * C * T[2], k0=2 * C * T[2] * T[1], f=2 * C * math.prod(T),
                 g=2 * C * F * math.prod(T))
        ws['filter'] = w['g'] * G
        if k[1] == 2:                                    # HWCF: rows c, cols f
            pre.append(Restage(kbuf, koff, 'filter', 0, (G, T[0], T[1], T[2]),
                               (k[0], k[2], k[3], k[4]), (w['g'], w['k0'], w['k1'], w['k2']),
                               C, F, k[5], w['f']))
        elif k[4] == 2 and k[3] == 2 * T[2] and k[2] == 2 * T[2] * T[1]:
            # FCHW with contiguous taps: rows c, cols = all taps of one filter.
            pre.append(Restage(kbuf, koff, 'filter', 0, (G, F, 1, 1), (k[0], k[1], 0, 0),
                               (w['g'], w['f'], 0, 0), C, math.prod(T), k[5], w['k2']))
        elif k[4] == 2:                                  # rows c, cols k2
            pre.append(Restage(kbuf, koff, 'filter', 0, (G, F, T[0], T[1]),
                               (k[0], k[1], k[2], k[3]), (w['g'], w['f'], w['k0'], w['k1']),
                               C, T[2], k[5], w['k2']))
        else:
            return scalar
        p['filter'] = ('filter', 0, (w['g'], w['f'], w['k0'], w['k1'], w['k2'], w['c']))

    # Output -> [N G O0 O1 O2 F], F contiguous; restaged in, accumulated, restaged out.
    if o[5] != 2:
        if o[4] != 2:
            return scalar
        w = dict(f=2, o2=2 * F, o1=2 * F * O[2], o0=2 * F * O[2] * O[1], g=2 * F * math.prod(O))
        w['n'] = w['g'] * G
        ws['output'] = w['n'] * N
        pre.append(Restage(obuf, ooff, 'output', 0, (N, G, O[0], O[1]), (o[0], o[1], o[2], o[3]),
                           (w['n'], w['g'], w['o0'], w['o1']), F, O[2], o[5], w['o2']))
        post.append(Restage('output', 0, obuf, ooff, (N, G, O[0], O[1]),
                            (w['n'], w['g'], w['o0'], w['o1']), (o[0], o[1], o[2], o[3]),
                            O[2], F, w['o2'], o[5]))
        p['output'] = ('output', 0, (w['n'], w['g'], w['o0'], w['o1'], w['o2'], w['f']))

    # Fold k2 into the reduction when (k2, c) is one contiguous run in both the
    # input window and the filter: C' = K2*C, K2' = 1. Windows of neighbouring
    # pixels then overlap, which one SDMA cannot express (row pitch >= row
    # width), so the kernel moves pixel rows in K2 residue classes; fold only
    # when that saves reduction tiles (C not already a multiple of 32).
    ii, kk = p['input'][2], p['filter'][2]
    tiles = lambda n: -(-n // TILE)
    if (T[2] > 1 and p['dilations'][2] == 1 and ii[5] == 2 and kk[5] == 2
            and ii[4] == 2 * C and kk[4] == 2 * C and tiles(T[2] * C) < T[2] * tiles(C)):
        e = dict(e, C=C * T[2], K2=1)
        p['extents'] = e
        p['fold'] = T[2]

    # SDMA row pitches of the compute loop and of every restage.
    kf = p.get('fold', 1)
    pitches = [p['input'][2][4] * p['steps'][2] * kf, p['filter'][2][1], p['output'][2][4] * kf]
    pitches += [x for r in pre + post for x in (r.src_row, r.dst_row)]
    if not all(_pitch_ok(x) for x in pitches):
        return scalar
    return ConvPlan(path='array', pre=tuple(pre), post=tuple(post), workspaces=ws, **p)
