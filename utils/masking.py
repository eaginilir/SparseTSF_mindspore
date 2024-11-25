import mindspore
from mindspore import Tensor
import numpy as np
import math


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = (B, 1, L, L)
        self._mask = Tensor(np.triu(np.ones(mask_shape, dtype=bool), k=1), mindspore.bool_)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = Tensor(np.triu(np.ones((L, scores.shape[-1]), dtype=bool), k=1), mindspore.bool_)
        _mask_ex = _mask.expand_dims(0).expand_dims(0).broadcast_to((B, H, L, scores.shape[-1]))
        batch_indices = Tensor(np.arange(B)[:, None, None], mindspore.int32)
        head_indices = Tensor(np.arange(H)[None, :, None], mindspore.int32)
        indicator = _mask_ex[batch_indices, head_indices, index]
        self._mask = indicator.reshape(scores.shape)

    @property
    def mask(self):
        return self._mask


class LocalMask():
    def __init__(self, B, L, S, device="cpu"):
        mask_shape = (B, 1, L, S)
        self.len = math.ceil(np.log2(L))
        mask1 = Tensor(np.triu(np.ones(mask_shape, dtype=bool), k=1), mindspore.bool_)
        mask2 = ~Tensor(np.triu(np.ones(mask_shape, dtype=bool), k=-self.len), mindspore.bool_)
        self._mask = mask1 | mask2

    @property
    def mask(self):
        return self._mask

