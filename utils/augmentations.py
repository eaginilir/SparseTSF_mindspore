import mindspore
from mindspore import Tensor, ops
import numpy as np
import mindspore.numpy as mnp


def augmentation(augment_time):
    if augment_time == 'batch':
        return BatchAugmentation()
    elif augment_time == 'dataset':
        return DatasetAugmentation()


class BatchAugmentation():
    def __init__(self):
        pass

    def freq_mask(self, x, y, rate=0.5, dim=1):
        xy = ops.concat([x, y], axis=dim)
        xy_f = mnp.fft.rfft(xy, axis=dim)
        m = Tensor(np.random.uniform(0, 1, xy_f.shape) < rate, mindspore.bool_)
        freal = ops.masked_fill(xy_f.real, m, 0)
        fimag = ops.masked_fill(xy_f.imag, m, 0)
        xy_f = mnp.fft.rfft(freal + 1j * fimag, axis=dim)
        xy = mnp.fft.irfft(xy_f, axis=dim)
        return xy

    def freq_mix(self, x, y, rate=0.5, dim=1):
        xy = ops.concat([x, y], axis=dim)
        xy_f = mnp.fft.rfft(xy, axis=dim)
        m = Tensor(np.random.uniform(0, 1, xy_f.shape) < rate, mindspore.bool_)
        amp = ops.abs(xy_f)
        _, index = ops.sort(amp, axis=dim, descending=True)
        dominant_mask = index > 2
        m = ops.logical_and(m, dominant_mask)
        freal = ops.masked_fill(xy_f.real, m, 0)
        fimag = ops.masked_fill(xy_f.imag, m, 0)

        b_idx = np.arange(x.shape[0])
        np.random.shuffle(b_idx)
        x2, y2 = x[b_idx], y[b_idx]
        xy2 = ops.concat([x2, y2], axis=dim)
        xy2_f = mnp.fft.rfft(xy2, axis=dim)

        m = ops.logical_not(m)
        freal2 = ops.masked_fill(xy2_f.real, m, 0)
        fimag2 = ops.masked_fill(xy2_f.imag, m, 0)

        freal += freal2
        fimag += fimag2

        xy_f = mnp.fft.irfft(freal + 1j * fimag, axis=dim)
        return xy

    def noise(self, x, y, rate=0.05, dim=1):
        xy = ops.concat([x, y], axis=1)
        noise_xy = (mnp.random.uniform(-0.5, 0.5, xy.shape) * 0.1).astype(mindspore.float32)
        xy = xy + noise_xy
        return xy

    def vFlip(self, x, y, rate=0.05, dim=1):
        xy = ops.concat([x, y], axis=1)
        xy = -xy
        return xy

    def hFlip(self, x, y, rate=0.05, dim=1):
        xy = ops.concat([x, y], axis=1)
        xy = ops.reverse_sequence(xy, seq_axis=dim)
        return xy

    def linear_upsampling(self, x, y, rate=0.5, dim=1):
        xy = ops.concat([x, y], axis=dim)
        original_shape = xy.shape
        start_point = np.random.randint(0, original_shape[1] // 2)

        xy = xy[:, start_point:start_point + original_shape[1] // 2, :]
        xy = xy.transpose(0, 2, 1)
        resize_op = ops.ResizeBilinear((original_shape[1], original_shape[2]))
        xy = resize_op(xy)
        xy = xy.transpose(0, 2, 1)
        return xy


class DatasetAugmentation():
    def __init__(self):
        pass

    def freq_dropout(self, x, y, dropout_rate=0.2, dim=0, keep_dominant=True):
        x, y = Tensor(x, mindspore.float32), Tensor(y, mindspore.float32)

        xy = ops.concat([x, y], axis=0)
        xy_f = mnp.fft.rfft(xy, axis=0)

        m = Tensor(np.random.uniform(0, 1, xy_f.shape) < dropout_rate, mindspore.bool_)

        freal = ops.masked_fill(xy_f.real, m, 0)
        fimag = ops.masked_fill(xy_f.imag, m, 0)
        xy_f = mnp.fft.irfft(freal + 1j * fimag, axis=dim)

        x, y = xy[:x.shape[0], :], xy[-y.shape[0]:, :]
        return x.asnumpy(), y.asnumpy()

    def freq_mix(self, x, y, x2, y2, dropout_rate=0.2):
        x, y = Tensor(x, mindspore.float32), Tensor(y, mindspore.float32)

        xy = ops.concat([x, y], axis=0)
        xy_f = mnp.fft.rfft(xy, axis=0)
        m = Tensor(np.random.uniform(0, 1, xy_f.shape) < dropout_rate, mindspore.bool_)
        amp = ops.abs(xy_f)
        _, index = ops.sort(amp, axis=0, descending=True)
        dominant_mask = index > 2
        m = ops.logical_and(m, dominant_mask)
        freal = ops.masked_fill(xy_f.real, m, 0)
        fimag = ops.masked_fill(xy_f.imag, m, 0)

        x2, y2 = Tensor(x2, mindspore.float32), Tensor(y2, mindspore.float32)
        xy2 = ops.concat([x2, y2], axis=0)
        xy2_f = mnp.fft.rfft(xy2, axis=0)

        m = ops.logical_not(m)
        freal2 = ops.masked_fill(xy2_f.real, m, 0)
        fimag2 = ops.masked_fill(xy2_f.imag, m, 0)

        freal += freal2
        fimag += fimag2

        xy_f = mnp.fft.irfft(freal + 1j * fimag, axis=0)
        x, y = xy[:x.shape[0], :], xy[-y.shape[0]:, :]
        return x.asnumpy(), y.asnumpy()
