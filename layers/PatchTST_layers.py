__all__ = ['Transpose', 'get_activation_fn', 'moving_avg', 'series_decomp', 'PositionalEncoding', 'SinCosPosEncoding', 'Coord2dPosEncoding', 'Coord1dPosEncoding', 'positional_encoding']           

import mindspore
import math
from mindspore import nn,ops
import mindspore.numpy as mnp
from mindspore.ops import operations as P
import numpy as np
from mindspore import Tensor

class Transpose(nn.Cell):
    def __init__(self, *dims, contiguous=False):
        super(Transpose, self).__init__()
        self.dims = dims
        self.contiguous = contiguous

    def construct(self, x):
        x = ops.transpose(x, self.dims)
        if self.contiguous:
            x = ops.Reshape()(x, x.shape)  # Ensure a new contiguous copy is created
        return x

    
def get_activation_fn(activation):
    if callable(activation): return activation()
    elif activation.lower() == "relu": return nn.ReLU()
    elif activation.lower() == "gelu": return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable') 
    
    
# decomposition

class moving_avg(nn.Cell):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, pad_mode='valid')

    def construct(self, x):
        # Padding on both ends of the time series
        front = ops.tile(x[:, 0:1, :], (1, (self.kernel_size - 1) // 2, 1))
        end = ops.tile(x[:, -1:, :], (1, (self.kernel_size - 1) // 2, 1))
        x = ops.concat([front, x, end], axis=1)
        x = self.avg(x.transpose(0, 2, 1))  # Change shape for AvgPool1d
        x = x.transpose(0, 2, 1)  # Restore original shape
        return x


class series_decomp(nn.Cell):
    """
    Series decomposition block that decomposes a time series into residual and moving average components.
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size)

    def construct(self, x):
        # Compute moving mean and residual
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return residual, moving_mean
    
    
    
# pos_encoding
def PositionalEncoding(q_len, d_model, normalize=True):
    pe = np.zeros((q_len, d_model), dtype=np.float32)
    position = np.arange(0, q_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    
    return Tensor(pe)

SinCosPosEncoding = PositionalEncoding

def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = 0.5 if exponential else 1
    for i in range(100):
        # Generate the coordinate positional encoding
        cpe = 2 * (np.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (np.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        
        if verbose:
            print(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}')
        
        if abs(cpe.mean()) <= eps:
            break
        elif cpe.mean() > eps:
            x += 0.001
        else:
            x -= 0.001
    
    # Normalize the positional encoding if needed
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    
    return Tensor(cpe, dtype=mindspore.float32)

def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    x = 0.5 if exponential else 1
    cpe = (2 * (np.linspace(0, 1, q_len).reshape(-1, 1) ** x) - 1)
    
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    
    return Tensor(cpe, dtype=mindspore.float32)

def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe is None:
        W_pos = np.random.uniform(-0.02, 0.02, (q_len, d_model)).astype(np.float32)
        learn_pe = False
    elif pe == 'zero':
        W_pos = np.random.uniform(-0.02, 0.02, (q_len, 1)).astype(np.float32)
    elif pe == 'zeros':
        W_pos = np.random.uniform(-0.02, 0.02, (q_len, d_model)).astype(np.float32)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = np.random.normal(0.0, 0.1, (q_len, 1)).astype(np.float32)
    elif pe == 'uniform':
        W_pos = np.random.uniform(0.0, 0.1, (q_len, 1)).astype(np.float32)
    elif pe == 'lin1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True).asnumpy()
    elif pe == 'exp1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True).asnumpy()
    elif pe == 'lin2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True).asnumpy()
    elif pe == 'exp2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True).asnumpy()
    elif pe == 'sincos':
        W_pos = PositionalEncoding(q_len, d_model, normalize=True).asnumpy()
    else:
        raise ValueError(f"{pe} is not a valid positional encoding type. Available types: 'gauss'=='normal', \
                            'zeros', 'zero', 'uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.")
    
    # Convert W_pos to MindSpore tensor
    W_pos_tensor = Tensor(W_pos, dtype=mindspore.float32)
    
    return nn.Parameter(W_pos_tensor, requires_grad=learn_pe)