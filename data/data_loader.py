# data/data_loader.py
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from config import ALL_COLUMNS, TARGET_COLUMNS, TARGET_INDICES, SCALING_CONFIG
from utils.tools import HybridScaler

class Dataset_MTS(Dataset):
    def __init__(self, root_path, data_path, flag='train', size=None, data_split=[0.7, 0.1, 0.2], scale=True):
        self.in_len = size[0]
        self.out_len = size[1]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.data_split = data_split
        self.__read_data__()

    def __read_data__(self):
        file_p = os.path.join(self.root_path, self.data_path)
        try:
            self.df_raw = pd.read_csv(file_p)
        except UnicodeDecodeError:
            self.df_raw = pd.read_csv(file_p, encoding='gbk')

        if '时间' in self.df_raw.columns:
            self.df_raw['时间'] = pd.to_datetime(self.df_raw['时间'])
            self.df_raw = self.df_raw.sort_values('时间').reset_index(drop=True)
        
        self.df_raw = self.df_raw.ffill().bfill()
        
        # 划分训练/验证/测试集
        if (self.data_split[0] > 1):
            train_num = int(self.data_split[0])
            val_num = int(self.data_split[1])
            test_num = int(self.data_split[2])
        else:
            train_num = int(len(self.df_raw) * self.data_split[0])
            test_num = int(len(self.df_raw) * self.data_split[2])
            val_num = len(self.df_raw) - train_num - test_num
        
        border1s = [0, train_num - self.in_len, train_num + val_num - self.in_len]
        border2s = [train_num, train_num + val_num, train_num + val_num + test_num]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        df_data = self.df_raw[ALL_COLUMNS]

        if self.scale:
            # 实例化混合缩放器
            self.scaler = HybridScaler(SCALING_CONFIG, ALL_COLUMNS, TARGET_COLUMNS)
            # 严格仅使用训练集部分来 fit
            self.scaler.fit(self.df_raw.iloc[0:train_num])
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        data = data.astype(np.float32)
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.in_len
        r_begin = s_end
        r_end = r_begin + self.out_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end][:, TARGET_INDICES] 
        return seq_x, seq_y
    
    def __len__(self):
        return len(self.data_x) - self.in_len - self.out_len + 1

    def inverse_transform(self, data, is_target=False):
        if self.scale:
            return self.scaler.inverse_transform(data, indices=TARGET_INDICES if is_target else None)
        return data