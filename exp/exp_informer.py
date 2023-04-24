from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息，不打印



class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            en_layers = self.args.en_layers if self.args.model=='informer' else self.args.stack_en_layers
            model = model_dict[self.args.model](
                self.args.en_input,
                self.args.de_input,
                self.args.out_size,
                self.args.train_len,
                self.args.delead_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                en_layers,
                self.args.de_layers,
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def Data_Extraction(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
            'LVMH':Dataset_Custom,
            'S68_4':Dataset_Custom,
            'GT': Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        if args.embed != 'timeF':
            timeenc = 0
        else:
            timeenc = 1

        if flag == 'test':
            shuffle_flag = False;
            drop_last = True;
            batch_size = args.batch_size;
            freq=args.freq
        elif flag=='pred':
            shuffle_flag = False;
            drop_last = False;
            batch_size = 1;
            freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True;
            drop_last = True;
            batch_size = args.batch_size;
            freq=args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.train_len, args.delead_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def select_modeloptim(self):
        if self.args.optimizer == 'adam':
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'adammax':
            #Adamax：Adamax对Adam进行了改进，使用了无限范数代替了二阶动量，可以在处理稀疏数据时表现更好。
            model_optim = optim.Adamax(self.model.parameters(), lr=self.args.learning_rate)
        #Adadelta：Adadelta类似于RMSprop，但使用了更复杂的学习率调整策略，可以更好地处理长期依赖问题
        #model_optim = optim.Adadelta(self.model.parameters(), lr=self.args.learning_rate)

        return model_optim

    def select_modelcriterion(self):
        if self.args.lossfunction == 'smoothL1':
            model_criterion = nn.SmoothL1Loss()
        elif self.args.lossfunction == 'mse':
            model_criterion = nn.MSELoss()  # 选择损失函数:MAE:
        elif self.args.lossfunction == 'L1':
            model_criterion = nn.L1Loss()
        return model_criterion

    def batch_process(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # 将 batch_x 和 batch_y 转换为浮点数类型，并将 batch_x 移动到指定的设备上
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()
        # 输出 batch_x 和 batch_y 的形状
        # 将 batch_x_mark 和 batch_y_mark 转换为浮点数类型，并将它们移动到指定的设备上
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        # 输出 batch_x_mark 和 batch_y_mark 的形状

        # decoder input

        if self.args.padding==0:
            de_input = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            de_input = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        de_input = torch.cat([batch_y[:,:self.args.delead_len,:], de_input], dim=1).float().to(self.device)
        # de_inp长度为72，其中前48为真实值，后面24个是要预测的值（用0初始化）
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, de_input, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, de_input, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, de_input, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, de_input, batch_y_mark)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)

        if self.args.features == 'MS':
            f_dim = -1
        else:
            f_dim = 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        return outputs, batch_y

    def train(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_data, train_loader = self.Data_Extraction(flag = 'train')
        vali_data, vali_loader = self.Data_Extraction(flag = 'val')
        test_data, test_loader = self.Data_Extraction(flag = 'test')
        model_optim = self.select_modeloptim()
        model_criterion = self.select_modelcriterion()
        batch_endtime = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        train_loss_store = []
        vali_loss_store = []
        test_loss_store = []

        for epoch in range(self.args.train_epochs):
            iter_num = 0
            train_loss = []
            
            self.model.train()  #模型设为训练模式
            epoch_strtime = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_num += 1
                
                model_optim.zero_grad()
                train_pred, real_data = self.batch_process(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = model_criterion(train_pred, real_data)
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    speed = (time.time()-batch_endtime)/iter_num
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    iter_num = 0
                    batch_endtime = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, model_criterion)
            test_loss = self.vali(test_data, test_loader, model_criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            train_loss_store.append(train_loss)
            vali_loss_store.append(vali_loss)
            test_loss_store.append(test_loss)
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args) #每个epoch结束之后自动调整学习率，变为原来的1/2
        
        #save train, validation loss
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'train_loss.npy', train_loss_store)
        np.save(folder_path+'vali_loss.npy', vali_loss_store)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def vali(self, vali_data, vali_loader, model_criterion):
        self.model.eval()
        total_vali_loss = []

        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            vali_pred, real_data = self.batch_process(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = model_criterion(vali_pred.detach().cpu(), real_data.detach().cpu())
            total_vali_loss.append(loss)
        total_vali_loss = np.average(total_vali_loss)
        self.model.train()
        return total_vali_loss

    def test(self, setting):
        self.model.eval()
        test_data, test_loader = self.Data_Extraction(flag='test')
        test_preds = []
        real_datas = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            test_pred, real_data = self.batch_process(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            test_preds.append(test_pred.detach().cpu().numpy())
            real_datas.append(real_data.detach().cpu().numpy())

        test_preds = np.array(test_preds)
        real_datas = np.array(real_datas)
        test_preds = test_preds.reshape(-1, test_preds.shape[-2], test_preds.shape[-1])
        real_datas = real_datas.reshape(-1, real_datas.shape[-2], real_datas.shape[-1])

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(test_preds, real_datas)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', test_preds)
        np.save(folder_path+'true.npy', real_datas)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self.Data_Extraction(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        pred_preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            pred_pred, real_data = self.batch_process(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            pred_preds.append(pred_pred.detach().cpu().numpy())

        pred_preds = np.array(pred_preds)
        pred_preds = pred_preds.reshape(-1, pred_preds.shape[-2], pred_preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', pred_preds)
        
        return

