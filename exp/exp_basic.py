import os
import mindspore
from mindspore import context,nn
import numpy as np


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            if not self.args.use_multi_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
            
            context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
            context.set_context(device_id=int(self.args.gpu))
            self.device = f"GPU:{self.args.gpu}"
            print(f'Use GPU: {self.device}')
        else:
            context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
            self.device = "CPU"
            print('Use CPU')
        
        return self.device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
