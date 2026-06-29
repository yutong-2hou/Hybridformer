# utils/tools.py
import numpy as np
import torch

class HybridScaler:
    def __init__(self, static_config, all_columns, dynamic_columns):
        self.static_config = static_config
        self.all_columns = all_columns
        self.dynamic_columns = dynamic_columns 
        self.mins = None
        self.maxs = None
        self.range = None

    def fit(self, train_df):
        final_mins = []
        final_maxs = []
        
        for col in self.all_columns:
            if col in self.dynamic_columns:
                p_min = train_df[col].min()
                p_max = train_df[col].max()
            else:
                p_min, p_max = self.static_config.get(col, (0, 1.0))
            
            final_mins.append(p_min)
            final_maxs.append(p_max)
            
        self.mins = np.array(final_mins, dtype=np.float32)
        self.maxs = np.array(final_maxs, dtype=np.float32)
        self.range = self.maxs - self.mins
        self.range[self.range == 0] = 1.0 

    def transform(self, data):
        data_clipped = np.clip(data, self.mins, self.maxs)
        return (data_clipped - self.mins) / self.range
    
    def inverse_transform(self, data, indices=None):
        m = self.mins if indices is None else self.mins[indices]
        r = self.range if indices is None else self.range[indices]

        if torch.is_tensor(data):
            device = data.device
            m_ts = torch.from_numpy(m).to(device).float()
            r_ts = torch.from_numpy(r).to(device).float()
            return data * r_ts + m_ts
        else:
            return data * r + m

def metric(pred, true):
    mae = np.mean(np.abs(pred - true))
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((pred - true) / true))
    mspe = np.mean(np.square((pred - true) / true))
    return mae, mse, rmse, mape, mspe

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience, self.verbose, self.counter = patience, verbose, 0
        self.best_score, self.early_stop, self.val_loss_min = None, False, np.Inf
        self.delta = delta
    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    def save_checkpoint(self, val_loss, model, path):
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss