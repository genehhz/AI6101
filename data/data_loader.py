import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):

        if size is not None:
            self.train_len = size[0]
            self.delead_len = size[1]
            self.pred_len = size[2]
        else:
            self.train_len = 24 * 4 * 4
            self.delead_len = 24 * 4
            self.pred_len = 24 * 4

        # init
        assert flag in ['train', 'test', 'val']
        mode_list = {'train': 0, 'val': 1, 'test': 2}
        self.choose_mode = mode_list[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.Data_load()

    def Data_load(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        mode_startindex = [0, 12 * 30 * 24 - self.train_len, 12 * 30 * 24 + 4 * 30 * 24 - self.train_len]
        mode_endindex = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border_partstart = mode_startindex[self.choose_mode]
        border_partend = mode_endindex[self.choose_mode]

        if self.features == 'M' or self.features == 'MS':
            No_time_datacols = df_raw.columns[1:]
            df_data = df_raw[No_time_datacols]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            mode_train_dataset = df_data[mode_startindex[0]:mode_endindex[0]]
            self.scaler.fit(mode_train_dataset.values)
            scaled_data = self.scaler.transform(df_data.values)
        else:
            scaled_data = df_data.values

        # 将特征数据进行缩放处理，使其在相同的尺度范围内。
        time_data = df_raw[['date']][border_partstart:border_partend]
        time_data['date'] = pd.to_datetime(time_data.date)
        timedata_encode = time_features(time_data, timeenc=self.timeenc, freq=self.freq)

        self.data_cut = scaled_data[border_partstart:border_partend]
        if self.inverse:
            self.data_cut_trans = df_data.values[border_partstart:border_partend]
        else:
            self.data_cut_trans = scaled_data[border_partstart:border_partend]
        self.timedata_encode = timedata_encode

    def __getitem__(self, index):
        dataseq_start = index
        dataseq_end = dataseq_start + self.train_len
        decoder_start = dataseq_end - self.delead_len
        decoder_end = decoder_start + self.delead_len + self.pred_len
        dataseq_input = self.data_cut[dataseq_start:dataseq_end]

        if self.inverse:
            dataseq_output = np.concatenate([self.data_cut[decoder_start:decoder_start + self.delead_len],
                                             self.data_cut_trans[decoder_start + self.delead_len:decoder_end]],
                                            0)
        else:
            dataseq_output = self.data_cut_trans[decoder_start:decoder_end]

        dataseq_input_mark = self.timedata_encode[dataseq_start:dataseq_end]
        dataseq_output_mark = self.timedata_encode[decoder_start:decoder_end]

        return dataseq_input, dataseq_output, dataseq_input_mark, dataseq_output_mark

    def __len__(self):
        return len(self.data_cut) - self.train_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):

        if size is not None:
            self.train_len = size[0]
            self.delead_len = size[1]
            self.pred_len = size[2]
        else:
            self.train_len = 24 * 4 * 4
            self.delead_len = 24 * 4
            self.pred_len = 24 * 4
        # init
        assert flag in ['train', 'test', 'val']
        mode_list = {'train': 0, 'val': 1, 'test': 2}
        self.choose_mode = mode_list[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.Data_load()

    def Data_load(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        mode_startindex = [0, 12 * 30 * 24 * 4 - self.train_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.train_len]
        mode_endindex = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border_partstart = mode_startindex[self.choose_mode]
        border_partend = mode_endindex[self.choose_mode]

        if self.features == 'M' or self.features == 'MS':
            No_time_datacols = df_raw.columns[1:]
            df_data = df_raw[No_time_datacols]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # 将特征数据进行缩放处理，使其在相同的尺度范围内。
        if self.scale:
            mode_train_dataset = df_data[mode_startindex[0]:mode_endindex[0]]
            self.scaler.fit(mode_train_dataset.values)
            scaled_data = self.scaler.transform(df_data.values)
        else:
            scaled_data = df_data.values

        time_data = df_raw[['date']][border_partstart:border_partend]
        time_data['date'] = pd.to_datetime(time_data.date)
        timedata_encode = time_features(time_data, timeenc=self.timeenc, freq=self.freq)

        self.data_cut = scaled_data[border_partstart:border_partend]
        if self.inverse:
            self.data_cut_trans = df_data.values[border_partstart:border_partend]
        else:
            self.data_cut_trans = scaled_data[border_partstart:border_partend]
        self.timedata_encode = timedata_encode

    def __getitem__(self, index):
        dataseq_start = index
        dataseq_end = dataseq_start + self.train_len
        decoder_start = dataseq_end - self.delead_len
        decoder_end = decoder_start + self.delead_len + self.pred_len

        dataseq_input = self.data_cut[dataseq_start:dataseq_end]
        if self.inverse:
            dataseq_output = np.concatenate([self.data_cut[decoder_start:decoder_start + self.delead_len],
                                             self.data_cut_trans[decoder_start + self.delead_len:decoder_end]],
                                            0)
        else:
            dataseq_output = self.data_cut_trans[decoder_start:decoder_end]

        dataseq_input_mark = self.timedata_encode[dataseq_start:dataseq_end]
        dataseq_output_mark = self.timedata_encode[decoder_start:decoder_end]

        return dataseq_input, dataseq_output, dataseq_input_mark, dataseq_output_mark

    def __len__(self):
        return len(self.data_cut) - self.train_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):

        if size is not None:
            self.train_len = size[0]
            self.delead_len = size[1]
            self.pred_len = size[2]
        else:
            self.train_len = 24 * 4 * 4
            self.delead_len = 24 * 4
            self.pred_len = 24 * 4

        # init
        assert flag in ['train', 'test', 'val']
        mode_list = {'train': 0, 'val': 1, 'test': 2}
        self.choose_mode = mode_list[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.Data_load()

    def Data_load(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        train_size = int(len(df_raw) * 0.7)
        test_size = int(len(df_raw) * 0.2)
        vali_size = len(df_raw) - train_size - test_size
        mode_startindex = [0, train_size - self.train_len, len(df_raw) - test_size - self.train_len]
        mode_endindex = [train_size, train_size + vali_size, len(df_raw)]
        border_partstart = mode_startindex[self.choose_mode]
        border_partend = mode_endindex[self.choose_mode]

        if self.features == 'M' or self.features == 'MS':
            No_time_datacols = df_raw.columns[1:]
            df_data = df_raw[No_time_datacols]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # 将特征数据进行缩放处理，使其在相同的尺度范围内。
        if self.scale:
            mode_train_dataset = df_data[mode_startindex[0]:mode_endindex[0]]
            self.scaler.fit(mode_train_dataset.values)
            scaled_data = self.scaler.transform(df_data.values)
        else:
            scaled_data = df_data.values

        time_data = df_raw[['date']][border_partstart:border_partend]
        time_data['date'] = pd.to_datetime(time_data.date)
        timedata_encode = time_features(time_data, timeenc=self.timeenc, freq=self.freq)

        self.data_cut = scaled_data[border_partstart:border_partend]
        if self.inverse:
            self.data_cut_trans = df_data.values[border_partstart:border_partend]
        else:
            self.data_cut_trans = scaled_data[border_partstart:border_partend]
        self.timedata_encode = timedata_encode

    def __getitem__(self, index):
        dataseq_start = index
        dataseq_end = dataseq_start + self.train_len
        decoder_start = dataseq_end - self.delead_len
        decoder_end = decoder_start + self.delead_len + self.pred_len

        dataseq_input = self.data_cut[dataseq_start:dataseq_end]
        if self.inverse:
            dataseq_output = np.concatenate([self.data_cut[decoder_start:decoder_start + self.delead_len],
                                             self.data_cut_trans[decoder_start + self.delead_len:decoder_end]],
                                            0)
        else:
            dataseq_output = self.data_cut_trans[decoder_start:decoder_end]
        dataseq_input_mark = self.timedata_encode[dataseq_start:dataseq_end]
        dataseq_output_mark = self.timedata_encode[decoder_start:decoder_end]

        return dataseq_input, dataseq_output, dataseq_input_mark, dataseq_output_mark

    def __len__(self):
        return len(self.data_cut) - self.train_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):

        if size is not None:
            self.train_len = size[0]
            self.delead_len = size[1]
            self.pred_len = size[2]
        else:
            self.train_len = 24 * 4 * 4
            self.delead_len = 24 * 4
            self.pred_len = 24 * 4

        # init
        assert flag in ['pred']
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.Data_load()

    def Data_load(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        border_partstart = len(df_raw) - self.train_len
        border_partend = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            No_time_datacols = df_raw.columns[1:]
            df_data = df_raw[No_time_datacols]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # 将特征数据进行缩放处理，使其在相同的尺度范围内。
        if self.scale:
            self.scaler.fit(df_data.values)
            scaled_data = self.scaler.transform(df_data.values)
        else:
            scaled_data = df_data.values

        past_timedata = df_raw[['date']][border_partstart:border_partend]
        past_timedata['date'] = pd.to_datetime(past_timedata.date)
        pred_timedata = pd.date_range(past_timedata.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        Sum_timedata = pd.DataFrame(columns=['date'])
        Sum_timedata.date = list(past_timedata.date.values) + list(pred_timedata[1:])
        timedata_encode = time_features(Sum_timedata, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_cut = scaled_data[border_partstart:border_partend]
        if self.inverse:
            self.data_cut_trans = df_data.values[border_partstart:border_partend]
        else:
            self.data_cut_trans = scaled_data[border_partstart:border_partendborder_partend]
        self.timedata_encode = timedata_encode

    def __getitem__(self, index):
        dataseq_start = index
        dataseq_end = dataseq_start + self.train_len
        decoder_start = dataseq_end - self.delead_len
        decoder_end = decoder_start + self.delead_len + self.pred_len

        dataseq_input = self.data_cut[dataseq_start:dataseq_end]
        if self.inverse:
            dataseq_output = self.data_cut[decoder_start:decoder_start + self.delead_len]
        else:
            dataseq_output = self.data_cut_trans[decoder_start:decoder_start + self.delead_len]

        dataseq_input_mark = self.timedata_encode[dataseq_start:dataseq_end]
        dataseq_output_mark = self.timedata_encode[decoder_start:decoder_end]

        return dataseq_input, dataseq_output, dataseq_input_mark, dataseq_output_mark

    def __len__(self):
        return len(self.data_cut) - self.train_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
