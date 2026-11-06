"""Dedicated depthwise kernels against independent BF16/i32 loop goldens."""
import itertools
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph.layout import TensorView
from graph.depthwise import DEPTHWISE_OPS, DepthwiseOp, lower_depthwise
from codegen.depthwise import emit_depthwise
from codegen.c_emitter import compile_and_assemble, render_in_file
from run_graph import _run_emulator


def encode(x, dtype='bf16'):
    if dtype == 'bf16':
        return torch.tensor(np.asarray(x).copy()).bfloat16().view(torch.int16).numpy().astype('<i2').tobytes()
    return np.asarray(x, dtype='<i4').tobytes()


def decode(x, dtype='bf16'):
    if dtype == 'bf16':
        return (np.frombuffer(x, dtype='<u2').astype(np.uint32) << 16).view(np.float32)
    return np.frombuffer(x, dtype='<i4')


@pytest.fixture(scope='module')
def compiled(tmp_path_factory):
    programs = {}
    def compile_for(emission, name):
        if name not in programs:
            work = tmp_path_factory.mktemp(name)
            compile_and_assemble(emission, str(work), 'depthwise')
            programs[name] = emission.instr_text
        return programs[name]
    return compile_for


def execute(op, buffers, compiled, tmp_path, metrics=None):
    packed = emit_depthwise(op, buffers)
    packed.emission.instr_text = compiled(packed.emission, op.name)
    image = tmp_path/'depthwise.in'
    image.write_text(render_in_file(packed.emission))
    memory, unit = _run_emulator(str(image), str(tmp_path), 'depthwise')
    if metrics is not None:
        metrics.update(unit.perf_metrics.metrics)
    return {name: bytes((memory.read_data((base+i)&-4) >> (8*((base+i)&3))) & 255
                         for i in range(packed.buffer_sizes[name]))
            for name, base in packed.buffer_addresses.items()}


def golden(x, weights, initial, layout, spatial_rank, multiplier, steps, dilations, dtype):
    # Normalize only the independent golden's logical indices to N,spatial,C[,M].
    x = np.moveaxis(x, 1, -1) if layout[1] == 'C' else x
    weights = np.moveaxis(weights, 0, -1) if layout[1] == 'C' else weights
    y = np.moveaxis(initial, 1, -1).copy() if layout[1] == 'C' else initial.copy()
    kernel = weights.shape[:spatial_rank]
    for index in np.ndindex(y.shape):
        n, spatial = index[0], index[1:1+spatial_rank]
        channel = index[1+spatial_rank]
        m = index[-1] if multiplier else None
        acc = y[index]
        for k in itertools.product(*(range(size) for size in kernel)):
            position = tuple(p*s + j*d for p,s,j,d in zip(spatial, steps, k, dilations))
            windex = (*k, channel, m) if multiplier else (*k, channel)
            if dtype == 'bf16':
                product = torch.tensor(float(x[n, *position, channel]) * float(weights[windex])).bfloat16().float().item()
                acc = torch.tensor(float(acc) + product).bfloat16().float().item()
            else:
                bits = (int(acc) + int(x[n, *position, channel]) * int(weights[windex])) & 0xffffffff
                acc = bits if bits < (1 << 31) else bits - (1 << 32)
        y[index] = acc
    return np.moveaxis(y, -1, 1) if layout[1] == 'C' else y


@pytest.mark.parametrize('dtype', ['bf16', 'i32'])
@pytest.mark.parametrize('name', [
    'depthwise_conv_1d_ncw_cw', 'depthwise_conv_1d_nwc_wc', 'depthwise_conv_1d_nwc_wcm',
    'depthwise_conv_2d_nchw_chw', 'depthwise_conv_2d_nhwc_hwc', 'depthwise_conv_2d_nhwc_hwcm',
    'depthwise_conv_3d_ncdhw_cdhw', 'depthwise_conv_3d_ndhwc_dhwc', 'depthwise_conv_3d_ndhwc_dhwcm',
])
def test_named_depthwise_kernels(name, dtype, compiled, tmp_path):
    layout, rank, multiplier = DEPTHWISE_OPS[name]
    spatial, kernel, steps, dilations = (3,)*rank, (2,)*rank, (1,)*rank, (1,)*rank
    shape = (2, *spatial, 3)
    wshape = (*kernel, 3, 2) if multiplier else (*kernel, 3)
    yshape = (2, *((2,)*rank), 3, 2) if multiplier else (2, *((2,)*rank), 3)
    x = (np.arange(np.prod(shape)).reshape(shape)%7-3).astype(np.float32)
    w = (np.arange(np.prod(wshape)).reshape(wshape)%5-2).astype(np.float32)
    y = np.full(yshape, 2, np.float32)
    if layout[1] == 'C':
        x, w, y = np.moveaxis(x,-1,1), np.moveaxis(w,-1,0), np.moveaxis(y,-1,1)
    expected = golden(x,w,y,layout,rank,multiplier,steps,dilations,dtype)
    op = DepthwiseOp('linalg.'+name, TensorView('x',x.shape,dtype), TensorView('w',w.shape,dtype),
                     TensorView('y',y.shape,dtype), steps, dilations)
    buffers = {'x':encode(x,dtype),'w':encode(w,dtype),'y':encode(y,dtype)}
    result = execute(op,buffers,compiled,tmp_path)
    np.testing.assert_array_equal(decode(result['y'],dtype).reshape(y.shape),expected)
    assert result['x'] == buffers['x'] and result['w'] == buffers['w']


@pytest.mark.parametrize('name', list(DEPTHWISE_OPS))
def test_vector_tails_fractional_products_and_canaries(name, compiled, tmp_path):
    layout, rank, multiplier = DEPTHWISE_OPS[name]
    first = layout[1] == 'C'
    spatial = (3,)*(rank-1) + ((67,) if first else (3,))
    output_spatial = tuple(size-1 for size in spatial)
    channels, m = (2 if first or multiplier else 65), (65 if multiplier else 1)
    shape = (1, *spatial, channels)
    wshape = (*((2,)*rank), channels, m) if multiplier else (*((2,)*rank), channels)
    yshape = (1, *output_spatial, channels, m) if multiplier else (1, *output_spatial, channels)
    x = (np.arange(np.prod(shape)).reshape(shape)%17-8).astype(np.float32)/7
    w = (np.arange(np.prod(wshape)).reshape(wshape)%13-6).astype(np.float32)/9
    y = np.full(yshape, .125, np.float32)
    if first:
        x,w,y = np.moveaxis(x,-1,1),np.moveaxis(w,-1,0),np.moveaxis(y,-1,1)
    x,w = decode(encode(x)).reshape(x.shape),decode(encode(w)).reshape(w.shape)
    expected = golden(x,w,y,layout,rank,multiplier,(1,)*rank,(1,)*rank,'bf16')
    sentinel = b'\x35\x71'
    buffers = {key:sentinel+encode(value)+sentinel for key,value in [('x',x),('w',w),('y',y)]}
    op = DepthwiseOp(name,TensorView('x',x.shape,'bf16',offset_bytes=2),
                     TensorView('w',w.shape,'bf16',offset_bytes=2),
                     TensorView('y',y.shape,'bf16',offset_bytes=2))
    metrics = {}
    result = execute(op,buffers,compiled,tmp_path,metrics)
    assert result['x'] == buffers['x'] and result['w'] == buffers['w']
    assert result['y'] == sentinel+encode(expected)+sentinel
    assert metrics['dyn_retired_vector_alu'] > 0


@pytest.mark.parametrize('aligned', [True, False])
@pytest.mark.parametrize('case', ['product_round', 'sum_round', 'nan', 'infinity'])
def test_bf16_per_operation_rounding_and_special_values(aligned, case, compiled, tmp_path):
    values = {
        'product_round': ([1.0078125], [1.0078125], -1.015625, 0.),
        'sum_round': ([256.,1.,-256.], [1.,1.,1.], 0., 0.),
        'nan': ([np.nan], [1.], 1., np.nan),
        'infinity': ([np.inf], [0.], 1., np.nan),
    }
    x,w,initial,expected = values[case]
    offset = 2 if aligned else 1
    prefix = b'\x35'*offset
    suffix = b'\x71\x39'
    op = DepthwiseOp('depthwise_conv_1d_nwc_wc',
                     TensorView('x',(1,len(x),1),'bf16',offset_bytes=offset),
                     TensorView('w',(len(w),1),'bf16',offset_bytes=offset),
                     TensorView('y',(1,1,1),'bf16',offset_bytes=offset))
    buffers = {'x':prefix+encode(x)+suffix,'w':prefix+encode(w)+suffix,
               'y':prefix+encode([initial])+suffix}
    metrics = {}
    result = execute(op,buffers,compiled,tmp_path,metrics)
    bits = b'\xc0\x7f' if np.isnan(expected) else encode([expected])
    assert result['y'] == prefix+bits+suffix
    assert result['x'] == buffers['x'] and result['w'] == buffers['w']
    assert (metrics['dyn_retired_vector_alu'] > 0) == aligned


def test_i32_multiply_and_add_wrap_modulo32(compiled, tmp_path):
    x = np.array([2147483647,-2147483648],dtype=np.int32)
    w = np.array([2,-1],dtype=np.int32)
    op = DepthwiseOp('depthwise_conv_1d_nwc_wc',TensorView('x',(1,2,1),'i32'),
                     TensorView('w',(2,1),'i32'),TensorView('y',(1,1,1),'i32'))
    result = execute(op,{'x':encode(x,'i32'),'w':encode(w,'i32'),'y':encode([123],'i32')},compiled,tmp_path)
    expected = (int(x[0])*int(w[0])+int(x[1])*int(w[1])+123)&0xffffffff
    assert result['y'] == expected.to_bytes(4,'little')


@pytest.mark.parametrize('alias', ['source','weights','both'])
def test_aliases_snapshot_both_operands(alias, compiled, tmp_path):
    names = ('a','a','a') if alias == 'both' else (('a','w','a') if alias == 'source' else ('x','a','a'))
    source,weights,output = names
    prefix,suffix = b'\x35\x71',b'\x47\x13'
    x = np.array([1.,2.,3.,-4.,5.,6.],np.float32).reshape(1,3,2)
    w = np.array([2.,-1.,3.,2.],np.float32).reshape(2,2)
    buffers = {source:prefix+encode(x)+suffix}
    if weights not in buffers:
        buffers[weights] = prefix+encode(w)+suffix
    logical_w = decode(buffers[weights][2:10]).reshape(2,2)
    initial = decode(buffers[output][2:10]).reshape(1,2,2)
    expected = golden(x,logical_w,initial,'NWC',1,False,(1,),(1,),'bf16')
    op = DepthwiseOp('depthwise_conv_1d_nwc_wc',TensorView(source,x.shape,'bf16',offset_bytes=2),
                     TensorView(weights,w.shape,'bf16',offset_bytes=2),
                     TensorView(output,initial.shape,'bf16',offset_bytes=2))
    wanted = dict(buffers)
    patched = bytearray(wanted[output]);patched[2:10]=encode(expected);wanted[output]=bytes(patched)
    assert execute(op,buffers,compiled,tmp_path) == wanted


@pytest.mark.parametrize('empty_kernel', [True, False])
def test_empty_reductions_preserve_bits_and_empty_outputs_write_nothing(empty_kernel, compiled, tmp_path):
    op = DepthwiseOp('depthwise_conv_1d_nwc_wc',
                     TensorView('x',(1,0,2) if empty_kernel else (1,1,2),'bf16'),
                     TensorView('w',(0,2) if empty_kernel else (1,2),'bf16'),
                     TensorView('y',(1,3,2) if empty_kernel else (1,0,2),'bf16'))
    buffers = {'x':b'' if empty_kernel else encode([1.,2.]),
               'w':b'' if empty_kernel else encode([3.,4.]),
               'y':b'\xff\x7f\x00\x80'*3 if empty_kernel else b''}
    assert execute(op,buffers,compiled,tmp_path) == buffers


def test_rectangular_dilation_negative_strides_and_padded_output(compiled, tmp_path):
    x = (np.arange(1*2*4*6*8).reshape(1,2,4,6,8)%11-5).astype(np.float32)
    w = (np.arange(2*2*3*2).reshape(2,2,3,2)%5-2).astype(np.float32)
    yshape = (1,2,1,4,3)
    initial = np.ones(yshape,np.float32)
    expected = golden(x[:,:,:,::-1,:],w[:,:,:,::-1],initial,'NCDHW',3,False,(2,1,2),(2,1,3),'bf16')
    src = TensorView('x',x.shape,'bf16')
    strides = list(src.strides_bytes);strides[-2] *= -1
    filt = TensorView('w',w.shape,'bf16')
    fstrides = list(filt.strides_bytes);fstrides[-1] *= -1
    output_strides = TensorView('y',yshape,'i32').strides_bytes
    backing = bytearray(b'\x35\x71'+b'\xa7'*(np.prod(yshape)*4)+b'\x53\x19')
    wanted = bytearray(backing)
    for i in range(np.prod(yshape)):
        backing[2+i*4:4+i*4] = encode([1.])
        wanted[2+i*4:4+i*4] = encode(expected.ravel()[i:i+1])
    op = DepthwiseOp('depthwise_conv_3d_ncdhw_cdhw',
                     TensorView('x',x.shape,'bf16',tuple(strides),src.strides_bytes[-2]*5),
                     TensorView('w',w.shape,'bf16',tuple(fstrides),filt.strides_bytes[-1]),
                     TensorView('y',yshape,'bf16',output_strides,2),(2,1,2),(2,1,3))
    buffers = {'x':encode(x),'w':encode(w),'y':bytes(backing)}
    result = execute(op,buffers,compiled,tmp_path)
    assert result == dict(buffers,y=bytes(wanted))


@pytest.mark.parametrize('field,value,error,match', [
    ('name','depthwise_conv_2d_nhwc_hwc_q',NotImplementedError,'unsupported'),
    ('strides',(0,),ValueError,'strides'),
    ('strides',(1,1),ValueError,'strides'),
    ('dilations',(-1,),ValueError,'dilations'),
    ('strides',(1 << 31,),OverflowError,'signed 32'),
    ('weights',TensorView('w',(2,3),'bf16'),ValueError,'channels'),
    ('destination',TensorView('y',(2,2,2),'bf16'),ValueError,'batches'),
    ('destination',TensorView('y',(1,3,2),'bf16'),ValueError,'window'),
    ('destination',TensorView('y',(1,2,2),'bf16',(0,0,0)),ValueError,'overlap'),
    ('source',TensorView('x',(1,3,2),'f32'),NotImplementedError,'mixed dtype'),
])
def test_invalid_contracts(field, value, error, match):
    from dataclasses import replace
    op = DepthwiseOp('depthwise_conv_1d_nwc_wc',TensorView('x',(1,3,2),'bf16'),
                     TensorView('w',(2,2),'bf16'),TensorView('y',(1,2,2),'bf16'))
    with pytest.raises(error,match=match):
        lower_depthwise(replace(op,**{field:value}),{name:bytes(256) for name in ['x','w','y']})


def test_unsupported_same_dtype_and_multiplier_mismatch():
    op = DepthwiseOp('depthwise_conv_1d_nwc_wc',TensorView('x',(1,3,2),'f32'),
                     TensorView('w',(2,2),'f32'),TensorView('y',(1,2,2),'f32'))
    with pytest.raises(NotImplementedError,match='matching bf16 or i32'):
        lower_depthwise(op,{name:bytes(256) for name in ['x','w','y']})
    op = DepthwiseOp('depthwise_conv_1d_nwc_wcm',TensorView('x',(1,3,2),'bf16'),
                     TensorView('w',(2,2,3),'bf16'),TensorView('y',(1,2,2,2),'bf16'))
    with pytest.raises(ValueError,match='multiplier'):
        lower_depthwise(op,{name:bytes(256) for name in ['x','w','y']})


def test_emitter_checks_marker_address_budget(monkeypatch):
    import codegen.depthwise as packing
    monkeypatch.setattr(packing,'MAX_DATA_ADDRESS',0x1000+28)
    op = DepthwiseOp('depthwise_conv_1d_nwc_wc',TensorView('x',(1,3,2),'bf16'),
                     TensorView('w',(2,2),'bf16'),TensorView('y',(1,2,2),'bf16'))
    with pytest.raises(OverflowError,match='image marker'):
        emit_depthwise(op,{'x':bytes(12),'w':bytes(8),'y':bytes(8)})


def test_public_runtime_returns_actual_buffers(tmp_path):
    from depthwise_runtime import execute_depthwise
    op = DepthwiseOp('depthwise_conv_1d_nwc_wc',TensorView('x',(1,2,1),'bf16'),
                     TensorView('w',(2,1),'bf16'),TensorView('y',(1,1,1),'bf16'))
    result = execute_depthwise(op,{'x':encode([2.,3.]),'w':encode([4.,5.]),'y':encode([1.])},tmp_path)
    assert result.backend == 'emulator' and result.buffers['y'] == encode([24.])
    with pytest.raises(ValueError,match='filename'):
        execute_depthwise(op,{},tmp_path,tag='../escape')


@pytest.mark.parametrize('name', list(DEPTHWISE_OPS))
def test_all_layouts_asymmetric_geometry_and_strided_scalar_fallback(name, compiled, tmp_path):
    layout, rank, multiplier = DEPTHWISE_OPS[name]
    first = layout[1] == 'C'
    spatial = tuple(5+j for j in range(rank))
    kernel, steps, dilations = (2,)*rank, (2,)*rank, tuple(1+j%2 for j in range(rank))
    out_spatial = tuple((size-1-d)//2+1 for size,d in zip(spatial,dilations))
    shape = (1,*spatial,2)
    wshape = (*kernel,2,3) if multiplier else (*kernel,2)
    yshape = (1,*out_spatial,2,3) if multiplier else (1,*out_spatial,2)
    x = (np.arange(np.prod(shape)).reshape(shape)%7-3).astype(np.float32)
    w = (np.arange(np.prod(wshape)).reshape(wshape)%5-2).astype(np.float32)
    y = np.full(yshape,2,np.float32)
    if first:
        x,w,y = np.moveaxis(x,-1,1),np.moveaxis(w,-1,0),np.moveaxis(y,-1,1)
    source_axis,weight_axis = (2,1) if first else (1,0)
    sx = TensorView('x',x.shape,'bf16').strides_bytes
    sw = TensorView('w',w.shape,'bf16').strides_bytes
    source_strides,weight_strides = list(sx),list(sw)
    source_strides[source_axis] *= -1
    weight_strides[weight_axis] *= -1
    expected = golden(np.flip(x,source_axis),np.flip(w,weight_axis),y,layout,rank,multiplier,steps,dilations,'bf16')
    backing = bytearray(b'\x35'+(encode([2.])+b'\xa7\x19')*y.size+b'\x71\x39')
    wanted = bytearray(backing)
    for i in range(y.size):wanted[1+i*4:3+i*4] = encode(expected.ravel()[i:i+1])
    op = DepthwiseOp(name,
                     TensorView('x',x.shape,'bf16',tuple(source_strides),1+(x.shape[source_axis]-1)*sx[source_axis]),
                     TensorView('w',w.shape,'bf16',tuple(weight_strides),1+(w.shape[weight_axis]-1)*sw[weight_axis]),
                     TensorView('y',y.shape,'bf16',TensorView('y',y.shape,'i32').strides_bytes,1),steps,dilations)
    buffers = {'x':b'\x35'+encode(x)+b'\x71\x39','w':b'\x35'+encode(w)+b'\x71\x39','y':bytes(backing)}
    metrics = {}
    result = execute(op,buffers,compiled,tmp_path,metrics)
    assert result == dict(buffers,y=bytes(wanted))
    assert metrics['dyn_retired_vector_alu'] == 0


@pytest.mark.parametrize('aligned', [True, False])
@pytest.mark.parametrize('source_bits,initial_bits,expected', [
    (0x8000,0x8000,b'\x00\x80'),
    (0x7fff,0,b'\xc0\x7f'),
])
def test_bf16_signed_zero_and_raw_nan_payload(aligned, source_bits, initial_bits, expected, compiled, tmp_path):
    offset = 2 if aligned else 1
    prefix = b'\x35'*offset
    suffix = b'\x71\x39'
    op = DepthwiseOp('depthwise_conv_1d_nwc_wc',
                     TensorView('x',(1,1,1),'bf16',offset_bytes=offset),
                     TensorView('w',(1,1),'bf16',offset_bytes=offset),
                     TensorView('y',(1,1,1),'bf16',offset_bytes=offset))
    result = execute(op,{'x':prefix+source_bits.to_bytes(2,'little')+suffix,
                         'w':prefix+encode([1.])+suffix,
                         'y':prefix+initial_bits.to_bytes(2,'little')+suffix},compiled,tmp_path)
    assert result['y'] == prefix+expected+suffix
