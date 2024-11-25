import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as P
import mindspore.numpy as mnp
import numpy as np

class Model(nn.Cell):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.Linear = nn.Dense(in_channels=self.seq_len, out_channels=self.pred_len)
        # Use this line if you want to visualize the weights
        self.Linear_weight = mindspore.Parameter((1 / self.seq_len) * mnp.ones([self.pred_len, self.seq_len]), name="Linear_weight")


    def construct(self, x):
        # x: [Batch, Input length, Channel]
        x = self.Linear(P.Transpose()(x, (0,2,1,))).permute(0,2,1)
        return x # [Batch, Output length, Channel]