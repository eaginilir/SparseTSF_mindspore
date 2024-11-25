from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, PatchTST, SparseTSF
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import mindspore
from mindspore import context, nn, Tensor
from mindspore.train import Model
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore import ops
from mindspore.common import dtype as mstype
from mindspore import load_checkpoint, load_param_into_net
from mindspore.amp import auto_mixed_precision

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class OneCycleLR(LearningRateSchedule):
    def __init__(self, optimizer, max_lr, steps_per_epoch, epochs, pct_start=0.3, div_factor=25.0, final_div_factor=1e4):
        super(OneCycleLR, self).__init__()
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = steps_per_epoch * epochs
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor


        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor

        self.peak_step = int(self.total_steps * pct_start)
        self.anneal_step = self.total_steps - self.peak_step

    def construct(self, global_step):
        if global_step <= self.peak_step:
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * (global_step / self.peak_step)
        else:
            step = global_step - self.peak_step
            lr = self.max_lr - (self.max_lr - self.final_lr) * (step / self.anneal_step)
        return Tensor(lr, dtype=mstype.float32)


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'SparseTSF': SparseTSF
        }
        model = model_dict[self.args.model].Model(self.args)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = nn.Adam(self.model.trainable_params(), learning_rate=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == "mae":#返回绝对平均误差
            criterion = nn.L1Loss()
        elif self.args.loss == "mse":#返回均方误差
            criterion = nn.MSELoss()
        elif self.args.loss == "smooth":#返回平滑L1损失
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.set_train(False)
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = Tensor(batch_x, mindspore.float32)
            batch_y = Tensor(batch_y, mindspore.float32)
            batch_x_mark = Tensor(batch_x_mark, mindspore.float32)
            batch_y_mark = Tensor(batch_y_mark, mindspore.float32)

            # decoder input
            dec_inp = ops.ZerosLike()(batch_y[:, -self.args.pred_len:, :])
            dec_inp = ops.Concat(1)((batch_y[:, :self.args.label_len, :], dec_inp))

            if 'Linear' in self.args.model or 'TST' in self.args.model or 'SparseTSF' in self.args.model:
                outputs = self.model(batch_x)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

            loss = criterion(outputs, batch_y)
            total_loss.append(loss.asnumpy())

        total_loss = np.average(total_loss)
        self.model.set_train(True)
        return total_loss#返回总损失

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        total_steps = train_steps * self.args.train_epochs  # 总步数

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = OneCycleLR(
            optimizer=model_optim,
            max_lr=self.args.learning_rate,
            steps_per_epoch=train_steps,
            epochs=self.args.train_epochs,
            pct_start=self.args.pct_start
        )

        model = Model(self.model, loss_fn=criterion, optimizer=model_optim, metrics=None)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.set_train(True)
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                batch_x = Tensor(batch_x, mindspore.float32)
                batch_y = Tensor(batch_y, mindspore.float32)
                batch_x_mark = Tensor(batch_x_mark, mindspore.float32)
                batch_y_mark = Tensor(batch_y_mark, mindspore.float32)

                # decoder input
                dec_inp = ops.ZerosLike()(batch_y[:, -self.args.pred_len:, :])
                dec_inp = ops.Concat(1)((batch_y[:, :self.args.label_len, :], dec_inp))

                if 'Linear' in self.args.model or 'TST' in self.args.model or 'SparseTSF' in self.args.model:
                    outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.asnumpy())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.asnumpy():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()
                    
                    
                if self.args.use_amp:
                    scaler = auto_mixed_precision.get_loss_scale_manager()
                    scaler.scale(loss).backward()
                    self.model.set_train()
                    scaler.update()
                else:
                    loss.backward()
                    self.model.set_train()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print(f'Updating learning rate to {scheduler.get_last_lr()[0]}')

        best_model_path = os.path.join(path, 'checkpoint.ckpt')
        param_dict = load_checkpoint(best_model_path)

        # 将加载的参数加载到模型中
        load_param_into_net(self.model, param_dict)

        return self.model

    def test(self, setting, test=0):
        # 加载数据集
        test_data, test_loader = self._get_data(flag='test')

        # 加载训练模型
        if test:
            print('loading model')
            checkpoint_path = os.path.join('./checkpoints/', setting, 'checkpoint.ckpt')
            checkpoint = load_checkpoint(checkpoint_path)
            load_param_into_net(self.model, checkpoint)

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 设置模型为评估模式
        self.model.set_train(False)
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = Tensor(batch_x, dtype=mstype.float32)
            batch_y = Tensor(batch_y, dtype=mstype.float32)

            batch_x_mark = Tensor(batch_x_mark, dtype=mstype.float32)
            batch_y_mark = Tensor(batch_y_mark, dtype=mstype.float32)

            # decoder input
            dec_inp = ops.Zeros()(batch_y[:, -self.args.pred_len:, :].shape, mstype.float32)
            dec_inp = ops.Concat(1)((batch_y[:, :self.args.label_len, :], dec_inp))

            # encoder - decoder 前向传播
            if self.args.use_amp:
                # 在MindSpore中没有直接支持AMP的API，但可以通过手动处理（如果需要）来优化
                if any(substr in self.args.model for substr in {'Linear', 'TST', 'SparseTSF'}):
                    outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if any(substr in self.args.model for substr in {'Linear', 'TST', 'SparseTSF'}):
                    outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

            outputs = outputs.asnumpy()  # 从计算图中分离，转换为NumPy数组
            batch_y = batch_y.asnumpy()

            # 储存预测结果
            preds.append(outputs)
            trues.append(batch_y)

            # 可视化，每20个批次的结果作为可视化的数据
            if i % 20 == 0:
                input_data = batch_x.asnumpy()
                gt = np.concatenate((input_data[0, :, -1], batch_y[0, :, -1]), axis=0)
                pd = np.concatenate((input_data[0, :, -1], outputs[0, :, -1]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # 计算模型的flop（浮点运算数）
        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1], batch_x.shape[2]))
            exit()

        # 处理预测结果并保存
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # 结果保存
        result_folder_path = './results/' + setting + '/'
        if not os.path.exists(result_folder_path):
            os.makedirs(result_folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        
        with open("result.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
            f.write('\n')
            f.write('\n')

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        # 加载训练好的模型
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.ckpt'
            checkpoint = load_checkpoint(best_model_path)
            load_param_into_net(self.model, checkpoint)

        # 模型预测
        preds = []

        self.model.set_train(False)  # 设置为评估模式
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            batch_x = Tensor(batch_x, dtype=mstype.float32)
            batch_y = Tensor(batch_y, dtype=mstype.float32)
            batch_x_mark = Tensor(batch_x_mark, dtype=mstype.float32)
            batch_y_mark = Tensor(batch_y_mark, dtype=mstype.float32)

            # decoder input
            dec_inp = ops.Zeros()(batch_y[:, -self.args.pred_len:, :].shape, mstype.float32)
            dec_inp = ops.Concat(1)((batch_y[:, :self.args.label_len, :], dec_inp))

            # 前向传播，生成预测
            if self.args.use_amp:
                # 目前MindSpore没有直接支持AMP的API，但可以手动处理
                if any(substr in self.args.model for substr in {'Linear', 'TST', 'SparseTSF'}):
                    outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if any(substr in self.args.model for substr in {'Linear', 'TST', 'SparseTSF'}):
                    outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            pred = outputs.asnumpy()  # 从计算图中分离，转换为NumPy数组
            preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # 结果保存
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print(folder_path)
        np.save(folder_path + 'real_prediction.npy', preds)

        return
