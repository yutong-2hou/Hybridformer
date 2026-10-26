# Hybridformer

面向细纱车间温湿度的多变量、多步时间序列预测研究原型。项目使用历史环境观测与设备状态共 **20 个输入变量**，预测未来 **6 个细纱温湿度变量**；默认用过去 **45 步**预测未来 **10 步**。

当前目标是建立可检查、可运行、可逐步复现的研究基线，再开展数据协议验证、模型检查与对比实验。模型入口类名为 `Crossformer`，实验类名为 `Exp_crossformer`；实现包含 RevIN、分段嵌入、时间/变量两阶段注意力和编码器—解码器。仓库尚未提供论文对应关系或严格复现报告，因此不将其声明为某篇论文的官方实现。

## 当前状态

- 已有：CSV 数据读取、混合缩放、训练/验证/测试、早停、权重和预测结果保存，以及一组历史实验产物。
- 本次整理仅新增文档与依赖清单，不改变现有 Python 代码、数据或权重。
- 待建立：自动化测试、随机种子与环境记录、独立推理入口、严格的数据和模型验证协议。
- 首个开发任务见 [Issue #1](https://github.com/yutong-2hou/Hybridformer/issues/1)，后续任务建议见 [开发路线](docs/ROADMAP.md)。

## 环境安装

建议使用 **Python 3.12** 和独立虚拟环境，在仓库根目录执行以下命令。`requirements.txt` 固定四项直接依赖，尚不等同于完整的跨平台环境锁文件。

```bash
git clone https://github.com/yutong-2hou/Hybridformer.git
cd Hybridformer
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python main.py --help
```

Windows 激活命令为 `.venv\Scripts\activate`。PyTorch 安装包需与操作系统及计算设备匹配；CUDA 环境应按 [PyTorch 官方历史版本安装说明的 v2.5.1 小节](https://pytorch.org/get-started/previous-versions/#v251)选择对应的安装包。依赖版本的本机验证范围见下方“验证范围”。

**兼容性约束：** 当前 `utils/tools.py` 使用 `np.Inf`，因此固定 `numpy==1.26.4`；直接升级到 NumPy 2 会导致早停初始化失败。`einops` 是模型分段和张量重排的必要依赖。

## 数据与任务约定

仓库附带 `dataset/AC4.csv`：UTF-8 编码、44,640 行、21 列，其中 `时间` 列加上 20 个输入变量；时间范围为 `2025/10/1 8:00` 至 `2025/11/1 7:59`。预测长度按行/步解释，代码不校验采样间隔或自动重采样。数据来源、各字段单位和再分发许可尚需维护者补充。

必需的输入列及读取顺序由 `config.py::ALL_COLUMNS` 确定：

```text
新风温度, 新风湿度,
细纱湿度1, 细纱湿度2, 细纱湿度3,
细纱温度1, 细纱温度2, 细纱温度3,
送风机1, 送风机2, 送风机3,
新风窗1, 新风窗2, 水泵1, 水泵2, 工艺排风窗,
二回风窗1, 二回风窗2, 回风机, 水池温度
```

输出顺序由 `TARGET_COLUMNS` 确定：`细纱湿度1, 细纱湿度2, 细纱湿度3, 细纱温度1, 细纱温度2, 细纱温度3`。替换 CSV 时需保留上述数值列；如果存在 `时间` 列，读取器按时间排序，否则沿用原始行顺序。当前未实现缺列、非数值和重复时间戳的友好校验。

读取器先在全表上执行 `ffill().bfill()`，然后按连续时间段划分数据。默认训练/验证/测试比例为 `0.7/0.1/0.2`：比例模式中训练和测试取整，验证使用剩余行数，第二个比例值并不直接决定验证行数。验证和测试允许使用前一段的 `in_len` 行作为历史上下文。六个目标变量的缩放范围只从训练段拟合，其余输入使用 `SCALING_CONFIG` 的固定范围；变换会裁剪到范围内，因此验证/测试目标也可能被裁剪。全表填充及裁剪对评估的影响仍需验证。

## 最小示例：不读取数据的模型前向计算

安装依赖后，在仓库根目录执行以下示例。它只检查 CPU 上的输入/输出接口，不训练、不写入权重，也不代表预测质量。

```python
import torch
from config import ALL_COLUMNS, TARGET_COLUMNS
from models.hybridformer import Crossformer

torch.manual_seed(0)
model = Crossformer(
    data_dim=len(ALL_COLUMNS), in_len=45, out_len=10, seg_len=5,
    win_size=2, factor=10, d_model=32, d_ff=64, n_heads=4,
    e_layers=2, dropout=0.0, device=torch.device("cpu"),
    c_out=len(TARGET_COLUMNS),
).eval()
x = torch.randn(2, 45, len(ALL_COLUMNS))
with torch.no_grad():
    y = model(x)
assert tuple(y.shape) == (2, 10, 6)
assert torch.isfinite(y).all()
print(tuple(y.shape))  # (2, 10, 6)
```

## 运行方式

### 1. 小规模训练与测试

以下命令使用 CSV 前 384 行，按 256/64/64 行划分，只训练 1 个 epoch。目的是检查完整流程，不能作为正式研究结果。实际滑动窗口数应为训练 202、验证 55、测试 55。

```bash
python main.py \
  --root_path ./ --data_path dataset/AC4.csv \
  --data_split 256,64,64 \
  --in_len 45 --out_len 10 --seg_len 5 \
  --data_dim 20 --c_out 6 \
  --d_model 32 --d_ff 64 --n_heads 4 --e_layers 2 \
  --batch_size 16 --train_epochs 1 --patience 1 \
  --checkpoints ./runs/smoke/
```

`data_split` 的首个值大于 1 时，三个值按行数解释；未使用的尾部数据不会进入该次实验。请确保训练段至少有 `in_len + out_len` 行，验证和测试段分别至少有 `out_len` 行，并且总行数不超过 CSV 长度。

### 2. 默认规模实验

```bash
python main.py \
  --root_path ./ --data_path dataset/AC4.csv \
  --checkpoints ./runs/default/
```

默认配置与历史 `args.json` 的主要参数一致：`in_len=45`、`out_len=10`、`seg_len=5`、`d_model=128`、`d_ff=256`、`n_heads=4`、`e_layers=2`、`dropout=0.2`、`batch_size=32`、`train_epochs=20`、`patience=4`、`learning_rate=0.0005`，`baseline=False`。

当前主入口总是执行 **训练 → 加载验证集损失最优权重 → 测试**，没有仅测试或仅推理的命令行开关。训练损失为缩放空间的 MSE，测试结果会反缩放后计算指标。

### 设备与参数注意事项

- 默认自动选择 `CUDA > MPS > CPU`；`--gpu` 仅设置 CUDA 设备编号。CPU 前向示例已显式选择 CPU。
- `--use_gpu` 和 `--scale` 使用 `type=bool`：传入字符串 `False` 仍会得到 `True`，不要使用 `--use_gpu False` 或 `--scale False` 来关闭功能。需要强制 CPU 的实验，可在 Python 调用 `Exp_crossformer` 前将 `args.use_gpu = False`；参数解析修复属于后续代码任务。
- 当前目标索引固定来自 `config.py`，不能只修改 `--data_dim` 或 `--c_out` 就切换数据任务。
- 实验子目录名只包含 `in_len/out_len/d_model/e_layers`，未包含全部参数或时间。同一输出根目录中的同名实验会覆盖原文件；每次正式实验请使用不同的 `--checkpoints` 路径。
- 仓库中已有 `checkpoints/` 历史产物；以上命令使用 `runs/` 避免覆盖它们。`runs/` 尚未被忽略，提交前需检查文件列表。

## 输出文件与结果解释

小规模命令的输出目录为 `runs/smoke/Hybrid_In45_Out10_dm32_el2/`，默认规模命令为 `runs/default/Hybrid_In45_Out10_dm128_el2/`：

| 文件 | 内容 |
| --- | --- |
| `args.json` | 本次运行参数 |
| `checkpoint.pth` | 验证损失最优的模型 `state_dict`，不包含完整环境和缩放器状态 |
| `metrics.txt` | 反缩放后的 MSE、MAE、RMSE、MAPE、MSPE |
| `pred.npy` / `true.npy` | 预测值与标签，形状为 `[测试窗口数, out_len, 6]` |

历史目录 `checkpoints/Hybrid_In45_Out10_dm128_el2/` 保存的记录为 MSE `0.09407378`、MAE `0.19786611`、RMSE `0.30671450`、MAPE `0.00428838`。这些是仓库已有记录，**本次没有复现该完整实验**，不能作为新环境可重复达到的承诺。

指标在所有窗口、预测步和六个目标上统一平均，混合了温度与湿度量纲，不能直接作为单目标的物理误差。MAPE/MSPE 是比例而非已经乘以 100 的百分数，且当前实现没有处理真实值为 0 的情况。正式比较还需要逐目标、逐预测步指标和统一的数据协议。

## 目录结构

```text
Hybridformer/
├── README.md                  # 项目入口与运行约定
├── requirements.txt           # 四项直接依赖的固定版本
├── main.py                    # 命令行参数；先训练再测试
├── config.py                  # 输入/目标列顺序及物理缩放范围
├── data/data_loader.py        # CSV、时间排序、划分与滑动窗口
├── dataset/AC4.csv            # 已提交的样例研究数据
├── models/hybridformer.py     # Crossformer 与相关模型层
├── exp/exp_hybridformer.py    # 设备、训练、验证、测试及产物保存
├── utils/tools.py            # HybridScaler、指标及早停
├── checkpoints/
│   └── Hybrid_In45_Out10_dm128_el2/  # 历史参数、权重、指标和数组
└── docs/ROADMAP.md            # 后续三个 Issue 建议及验收标准
```

仓库还跟踪了 `.DS_Store`、`__pycache__/` 和 `.pyc` 文件，暂未清理；当前也没有项目许可证。代码、数据和历史权重的许可应分别确认，公开可见不代表已授予再分发许可。

## 验证范围

文档基于原始提交 `b388e2515660d903011e01d57ac1d67ac9a3e935` 检查，核对了全部六个 Python 源文件、数据字段及历史参数/指标。2026-10-26 在 macOS 27.0.1 / Apple Silicon / Python 3.12.14 的独立环境中，使用 `requirements.txt` 的四个固定版本完成了以下验证：

- `pip check`、`python main.py --help` 通过；直接执行上述 CPU 前向示例得到 `(2, 10, 6)`，数值有限。
- 直接执行上述小规模训练命令，设备回退到 CPU；训练/验证/测试窗口数为 202/55/55，完成 1 个 epoch 并进入测试。
- 最优权重可重新加载；五个输出文件齐全，两个数组形状均为 `(55, 10, 6)`，数组和五项指标均为有限值。
- 六个原始 Python 文件与原始提交逐字节一致。另一次小模型反向诊断发现，最后一层编码器的 26 个参数均无梯度，已记录为后续模型核查任务。

小规模验证未固定训练随机种子，不应比较其单次损失。CUDA、MPS、跨平台与完整 20 epoch 历史指标复现不在本次验证范围。
