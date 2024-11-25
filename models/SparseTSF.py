import mindspore
from mindspore import nn, ops, Parameter
from mindspore.common import dtype as mstype
import mindspore.numpy as np

class Model(nn.Cell):
    def __init__(self, configs):
        super(Model, self).__init__()

        # get parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.d_model = configs.d_model
        self.model_type = configs.model_type
        assert self.model_type in ['linear', 'mlp']

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
                                stride=1, pad_mode="pad", padding=self.period_len // 2, has_bias=False)

        if self.model_type == 'linear':
            self.linear = nn.Dense(self.seg_num_x, self.seg_num_y, has_bias=False)
        elif self.model_type == 'mlp':
            self.mlp = nn.SequentialCell([
                nn.Dense(self.seg_num_x, self.d_model),
                nn.ReLU(),
                nn.Dense(self.d_model, self.seg_num_y)
            ])

    def construct(self, x):
        batch_size = x.shape[0]
        reduce_mean = ops.ReduceMean(keep_dims=True)
        seq_mean = reduce_mean(x, 1)
        x = (x - seq_mean).transpose(0, 2, 1)

        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x

        x = x.reshape(-1, self.seg_num_x, self.period_len).transpose(0, 2, 1)

        if self.model_type == 'linear':
            y = self.linear(x)
        elif self.model_type == 'mlp':
            y = self.mlp(x)

        y = y.transpose(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)

        y = y.transpose(0, 2, 1) + seq_mean

        return y
