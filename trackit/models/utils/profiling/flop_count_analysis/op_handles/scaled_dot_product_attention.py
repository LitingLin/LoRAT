import torch
import torch._C

def _prod(dims):
    p = 1
    for v in dims:
        p *= v
    return p

def scaled_dot_product_attention_flop_jit(inputs, outputs):
    q = inputs[0]
    k = inputs[1]
    v = inputs[2]
    q_shape = q.type().sizes()
    k_shape = k.type().sizes()
    v_shape = v.type().sizes()
    attn_mask = inputs[3]
    assert isinstance(attn_mask.type(), torch._C.NoneType), "flop_counter(sdpa): attn_mask currently not supported"
    is_causal = inputs[5]
    assert isinstance(is_causal.type(), torch._C.BoolType) and not is_causal.toIValue(), "flop_counter(sdpa): is_causal should be False, since causal attention is not supported"
    macs = _prod(q_shape) * k_shape[-2]
    macs += _prod(q_shape[:-1]) * k_shape[-2] * v_shape[-1]
    return macs
