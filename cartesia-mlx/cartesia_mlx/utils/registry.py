import math
import mlx.nn as nn
import mlx.core as mx

@mx.compile
def bert_gelu(x):
    return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * mx.power(x, 3.0))))


ACTIVATIONS = {
    "swish": nn.silu,
    "gelu": nn.gelu,
    "bert_gelu": bert_gelu,
}

NORMS = {
    "rms": nn.RMSNorm,
    "layer": nn.LayerNorm,
}