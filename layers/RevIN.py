import mindspore
from mindspore import nn,ops,Parameter
import numpy as np

class RevIN(nn.Cell):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def construct(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError("Mode should be 'norm' or 'denorm'.")
        return x

    def _init_params(self):
        # Initialize learnable parameters for affine transformation
        self.affine_weight = Parameter(ops.ones((self.num_features,), mindspore.float32))
        self.affine_bias = Parameter(ops.zeros((self.num_features,), mindspore.float32))

    def _get_statistics(self, x):
        dim_to_reduce = tuple(range(1, len(x.shape) - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = ops.ReduceMean(keep_dims=True)(x, dim_to_reduce).detach()
        self.stdev = ops.Sqrt()(
            ops.ReduceMean(keep_dims=True)((x - self.mean) ** 2, dim_to_reduce) + self.eps
        ).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x