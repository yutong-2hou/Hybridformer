# main.py
import argparse
import warnings
from exp.exp_hybridformer import Exp_crossformer

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HybridFormer Time Series Forecasting')
    
    # 路径与数据
    parser.add_argument('--root_path', type=str, default='./')
    # 请根据实际路径修改 data_path
    parser.add_argument('--data_path', type=str, default='dataset/AC4.csv')
    
    # 长度配置
    parser.add_argument('--in_len', type=int, default=45)
    parser.add_argument('--out_len', type=int, default=10)
    parser.add_argument('--seg_len', type=int, default=5)
    
    # 模型配置
    parser.add_argument('--data_dim', type=int, default=20)
    parser.add_argument('--c_out', type=int, default=6)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--win_size', type=int, default=2)
    parser.add_argument('--factor', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    # 训练配置
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    
    # 其他配置
    parser.add_argument('--scale', type=bool, default=True) 
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--data_split', type=str, default='0.7,0.1,0.2')
    parser.add_argument('--baseline', action='store_true', default=False)

    args = parser.parse_args()
    args.data_split = [float(i) for i in args.data_split.split(',')]

    setting = f'Hybrid_In{args.in_len}_Out{args.out_len}_dm{args.d_model}_el{args.e_layers}'
    
    exp = Exp_crossformer(args)
    
    print(f'>>>> Starting Training: {setting} <<<<')
    exp.train(setting)
    
    print(f'>>>> Starting Testing: {setting} <<<<')
    exp.test(setting)