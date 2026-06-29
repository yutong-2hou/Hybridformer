# exp/exp_hybridformer.py
import os
import time
import json
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np

from models.hybridformer import Crossformer
from data.data_loader import Dataset_MTS
from utils.tools import EarlyStopping, metric

class Exp_crossformer:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        
    def _acquire_device(self):
        """
        自动选择最佳计算设备：CUDA > MPS > CPU
        """
        if self.args.use_gpu:
            if torch.cuda.is_available():
                print(f'Use GPU: CUDA:{self.args.gpu}')
                return torch.device(f'cuda:{self.args.gpu}')
            # MPS 支持
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                print('Use GPU: MPS (MacOS Metal)')
                return torch.device('mps')
            else:
                print('GPU not found or not supported, falling back to CPU.')
        else:
            print('Use CPU')
        
        return torch.device('cpu')
    
    def _build_model(self):
        return Crossformer(self.args.data_dim, self.args.in_len, self.args.out_len, self.args.seg_len,
            self.args.win_size, self.args.factor, self.args.d_model, self.args.d_ff, self.args.n_heads, 
            self.args.e_layers, self.args.dropout, self.args.baseline, self.device, c_out=self.args.c_out)
    
    def _get_data(self, flag):
        data_set = Dataset_MTS(
            root_path=self.args.root_path, data_path=self.args.data_path, flag=flag,
            size=[self.args.in_len, self.args.out_len], data_split=self.args.data_split, scale=self.args.scale
        )
        data_loader = DataLoader(data_set, batch_size=self.args.batch_size, shuffle=(flag=='train'), num_workers=0)
        return data_set, data_loader
    
    def train(self, setting):
        train_data, train_loader = self._get_data('train')
        vali_data, vali_loader = self._get_data('val')
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path): 
            os.makedirs(path)
            
        with open(os.path.join(path, "args.json"), 'w') as f:
            json.dump(vars(self.args), f, indent=4)
            
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        criterion = nn.MSELoss()
        
        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss = []
            epoch_start_time = time.time()
            
            for i, (batch_x, batch_y) in enumerate(train_loader):
                model_optim.zero_grad()
                
                # 将数据移动到选定的设备 (CUDA/MPS/CPU)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                train_loss.append(loss.item())
                
                loss.backward()
                model_optim.step()
                
                if (i + 1) % 100 == 0:
                    cost_time = time.time() - epoch_start_time
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f} | cost_time: {cost_time:.2f}s")
                    
            epoch_end_time = time.time()
            vali_loss = self.vali(vali_loader, criterion)
            
            print(f"Epoch: {epoch + 1} done. Total cost: {epoch_end_time - epoch_start_time:.2f}s")
            print(f"Epoch: {epoch + 1} | Train Loss: {np.mean(train_loss):.7f} Vali Loss: {vali_loss:.7f}")
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop: 
                print("Early stopping triggered.")
                break
            
    def vali(self, loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for x, y in loader:
                x = x.float().to(self.device)
                y = y.float().to(self.device)
                
                pred = self.model(x)
                # 计算 Loss 时统一转回 CPU
                total_loss.append(criterion(pred.cpu(), y.cpu()).item())
        return np.average(total_loss)
    
    def test(self, setting):
        test_data, test_loader = self._get_data('test')
        
        best_model_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.float().to(self.device)
                y = y.float().to(self.device)

                pred = self.model(x).cpu()
                pred = test_data.inverse_transform(pred, is_target=True)
                true = test_data.inverse_transform(y.cpu(), is_target=True)
                
                preds.append(pred.numpy())
                trues.append(true.numpy())
                
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'Test Results -> MSE:{mse:.4f}, MAE:{mae:.4f}')
        
        folder_path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        with open(os.path.join(folder_path, "metrics.txt"), 'w') as f:
            f.write(f"mse:{mse}, mae:{mae}, rmse:{rmse}, mape:{mape}, mspe:{mspe}")
            
        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)
        
        return