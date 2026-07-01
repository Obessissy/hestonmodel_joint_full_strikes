# FinFlow — 面向区制切换市场的动作条件流匹配世界模型

本仓库是论文 **《面向区制切换市场的动作条件流匹配世界模型：从两阶段到联合转移流匹配，及一步 on-policy flow-map 蒸馏》**（[`paper/main_zh.tex`](paper/main_zh.tex)，编译版 [`paper/main_zh.pdf`](paper/main_zh.pdf)）的完整实现、模型权重与实验产物。

模型把市场视为一个 **动作条件世界模型**：外部智能体在每一步选择离散市场区制（正常 / 高波动 / 崩盘），驱动三状态马尔可夫切换 Heston 过程。模型学习单步转移核

```text
p_theta(log v_{t+1}, r_t | log v_t, r_{t-1}, a_t)
```

并在自由 rollout（无 teacher forcing）下自回归推演一个完整交易年（252 步）。原论文主指标是生成路径上欧式期权价格相对 10 万条 MC oracle 的定价 RMSE；当前版本进一步加入 **完整 strike surface、亚式期权定价误差和路径整体质量指标**，避免只看少数欧式期权点导致的评价偏窄。

## 当前评价协议更新

这次 README 与论文草稿中的更新对应以下变化：

- **更完整的 strike surface**：欧式与亚式期权都使用 17 个 moneyness × 3 个 maturity，共 51 个价格点；moneyness 为 `0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.30, 1.40, 1.50, 1.75, 2.00`，maturity 为 `0.25, 0.5, 1.0`。
- **新增亚式期权**：用路径平均价格的 payoff 定价，显式考察路径依赖信息；它不只依赖到期价格，因此能补充欧式期权主要约束终端分布的不足。
- **保留路径整体质量**：除期权 RMSE/MAPE 外，继续报告 marginal Wasserstein、total return Wasserstein、absolute total return Wasserstein 和 signature Wasserstein。
- **定价方式**：当前评估中的欧式 / 亚式价格均来自生成路径和 MC oracle 上的 Monte Carlo payoff 均值，不是重新求解 Feynman-Kac PDE；区别在于期权 payoff 是非线性函数，可以从不同 strike / maturity 上投影分布形状和尾部误差。
- **teacher 说明**：本次 full-surface evaluation 没有直接评估 joint-FM teacher 权重，因为本地 release 中缺少 `joint_fm_teacher/ema_epoch_060.pt`；on-policy 学生在训练时使用了 teacher endpoint，但当前评估脚本只扫 `release/checkpoints/**/best.pt` 的一步学生模型。

## 最新 full-surface 结果

评估输出位于：

- `runs/full_surface_eval/summary_full_surface.csv`
- `runs/full_surface_eval/summary_full_surface.json`

当前 13 个 release 学生权重中，新的综合最优模型是：

```text
release/checkpoints/onpolicy_flowmap/nfe120_h64_s30_e4/best.pt
```

| 模型 | 欧式 full-surface RMSE ↓ | 亚式 RMSE ↓ | Marginal W1 ↓ | Total-return W1 ↓ | Abs-return W1 ↓ | Signature W1 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| **On-policy flow-map, NFE120/h64/s30/e4** | **0.0917** | **0.0525** | 0.000299 | **0.00449** | 0.0437 | **0.00111** |
| On-policy flow-map, NFE120/h128/s30/e1（旧 BEST） | 0.1416 | 0.0787 | **0.000287** | 0.00534 | 0.0438 | 0.00129 |
| Distill flow-map naive | 0.2100 | 0.0932 | 0.000501 | 0.0152 | 0.1276 | 0.00400 |
| Distill CD | 0.2981 | 0.1347 | 0.000336 | 0.00986 | 0.0975 | 0.00203 |
| Distill MeanFlow | 1.6498 | 0.6282 | 0.001700 | 0.0344 | 0.4236 | 0.00793 |

注意：`nfe120_h128_s30_e1_BEST` 仍对应旧的较窄 15 点欧式协议下的交付命名；在完整 strike surface + 亚式期权 + 路径质量的评价下，`nfe120_h64_s30_e4` 更优。

## 方法主线

| 阶段 | 方法 | 论文章节 | 代码入口 |
|---|---|---|---|
| 基线 | Quant-GAN / GARCH(1,1)-t / 移动块自助 | 相关工作、实验 | `scripts/train_quant_gan.py`, `scripts/sample_quant_gan.py`, `analysis/remote_scripts/baseline_generate.py` |
| 第一阶段 | 两阶段转移流匹配 + 调度采样 | §4.1.1 | `scripts/train_vol_trans.py`, `scripts/train_ret_trans.py`, `scripts/train_transition_fm.py` |
| 路径微调探索 | SIGMA 路径分布微调 | §4.1.2 | `scripts/pathwise_teacher_combined.py`, `finflow/pathwise_teacher.py` |
| 定价微调反例 | 可微定价微调 | §4.1.3 | `scripts/finetune_flow_map_pricing.py` |
| **主方法 teacher** | **联合转移流匹配 teacher（joint-FM）** | §4.1.4 | `scripts/train_joint_trans.py`, `finflow/models/transition_fm.py` |
| 一步蒸馏对比 | CD / MeanFlow / flow-map | §4.2.1 | `scripts/distill_consistency.py`, `scripts/distill_mean_flow.py`, `scripts/distill_flow_map.py` |
| **主蒸馏** | **on-policy teacher-endpoint 修正** | §4.2.2 | `scripts/finetune_flow_map_onpolicy.py` |
| **完整评估** | **欧式 full surface + 亚式期权 + 路径质量** | 实验更新 | `scripts/evaluate_all_checkpoints_full_surface.ps1`, `scripts/evaluate_all_checkpoints_full_surface.cmd` |

## 目录结构

```text
finflow/                 核心包
  data/                  Heston QE 模拟、三状态区制切换、期权定价
  models/                TransitionFM、联合转移 FM、MeanFlow、Consistency
  distillation/          MeanFlow / Consistency / flow-map 蒸馏器
  pathwise_teacher.py    SIGMA 路径分布微调
  inference/             统一采样器与自回归 rollout
  eval/                  风格化事实、距离、定价、报告
  baselines/             Quant-GAN（TCN + Lambert-W + WGAN-GP）
scripts/                 CLI 入口；一个任务一个脚本
tests/                   pytest 测试套件
data/heston_v3/          数据集 splits、metadata、mc_oracle
analysis/                作图与可视化数据
paper/                   论文源码 main_zh.tex / PDF / references.bib
presentation/            汇报材料
release/                 交付模型权重与关键实验结果
runs/full_surface_eval/  当前完整 strike surface 与亚式期权评估输出
```

## 安装

```bash
pip install -r requirements.txt
```

主要依赖包括 `numpy`、`torch`、`tqdm`、`matplotlib`、`pytest`。

## 数据与 MC oracle

`generate_heston_data.py` 生成训练 / 验证 / 测试用的三状态区制切换 Heston 路径；`generate_mc_oracle.py` 生成更大样本的 oracle 路径，用来作为期权价格和分布距离的高精度参照。

```bash
python3 scripts/generate_heston_data.py \
  --output data/heston_v3 --n-train 50000 --n-val 5000 --n-test 10000 \
  --steps 252 --regimes --seed 1234

python3 scripts/generate_mc_oracle.py \
  --data-dir data/heston_v3 --output data/heston_v3/mc_oracle.npz --n-paths 100000
```

二者区别：前者是模型训练 / 验证 / 测试数据；后者不是训练数据，而是高样本数 Monte Carlo 基准，用于计算 oracle option price、Wasserstein、signature 等评估指标。

## 训练与蒸馏流程

```bash
# 1) 训练联合转移流匹配 teacher
python3 scripts/train_joint_trans.py \
  --data-dir data/heston_v3 --output-dir runs/joint_fm \
  --hidden-dim 512 --num-blocks 6 --batch-size 8192 --epochs 60 --lr 2e-4

# 2) 选择最优 EMA checkpoint / NFE
python3 scripts/select_joint_checkpoint.py \
  --checkpoints "runs/joint_fm/<run>/checkpoints/ema_epoch_*.pt" \
  --data-dir data/heston_v3 --mc-oracle data/heston_v3/mc_oracle.npz \
  --nfe-steps 120 --rank-by pricing_rmse --regime-actions \
  --output runs/joint_fm/selection.json

# 3) 一步蒸馏：flow-map；可对比 CD / MeanFlow
python3 scripts/distill_flow_map.py --stage joint \
  --teacher-checkpoint runs/joint_fm/<run>/checkpoints/ema_epoch_060.pt \
  --data-dir data/heston_v3 --output-dir runs/joint_distill --epochs 15 --batch-size 4096

# 4) on-policy teacher-endpoint 修正
python3 scripts/finetune_flow_map_onpolicy.py \
  --data-dir data/heston_v3 --mc-oracle data/heston_v3/mc_oracle.npz \
  --init-checkpoint runs/joint_distill/<run>/checkpoints/best.pt \
  --teacher-checkpoint runs/joint_fm/<run>/checkpoints/ema_epoch_060.pt \
  --output-dir runs/joint_onpolicy \
  --teacher-n-steps 120 --rollout-horizon 128 --path-batch-size 512 \
  --steps-per-epoch 30 --epochs 4 --lr 5e-6 --flowmap-weight 1
```

## 单模型评估

```bash
python3 scripts/rollout_joint.py \
  --checkpoint release/checkpoints/onpolicy_flowmap/nfe120_h64_s30_e4/best.pt \
  --data-dir data/heston_v3 \
  --output runs/rollout_onpolicy_h64_s30_e4.npz \
  --n-paths 10000 --n-steps 252 --regime-actions \
  --fm-n-steps 20 --fm-solver euler --device auto

python3 scripts/evaluate_rollout.py \
  --real data/heston_v3/test.npz \
  --fake runs/rollout_onpolicy_h64_s30_e4.npz \
  --data-dir data/heston_v3 \
  --mc-oracle data/heston_v3/mc_oracle.npz \
  --output runs/eval_onpolicy_h64_s30_e4_full_surface.json \
  --moneynesses 0.50 0.60 0.70 0.80 0.85 0.90 0.95 1.00 1.05 1.10 1.15 1.20 1.30 1.40 1.50 1.75 2.00 \
  --maturities 0.25 0.5 1.0 \
  --asian-moneynesses 0.50 0.60 0.70 0.80 0.85 0.90 0.95 1.00 1.05 1.10 1.15 1.20 1.30 1.40 1.50 1.75 2.00 \
  --asian-maturities 0.25 0.5 1.0 \
  --signature-depth 3
```

## 一键评估所有 release checkpoint

```cmd
set N_PATHS=10000
set DEVICE=auto
set FM_N_STEPS=20
set FM_SOLVER=euler
set FORCE=1
scripts\evaluate_all_checkpoints_full_surface.cmd
```

常用环境变量：

- `N_PATHS`：每个 checkpoint rollout 的路径数，默认 `10000`。
- `DEVICE`：`auto` / `cpu` / `cuda`。
- `FM_N_STEPS`：rollout 中 flow matching 求解步数，默认 `20`。
- `FORCE=1`：重新生成已有 rollout 和 evaluation；不设置时会复用已有文件。
- `LIMIT`：调试时限制评估使用的路径数。

PowerShell 也可直接运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\evaluate_all_checkpoints_full_surface.ps1" `
  -Python "C:\ProgramData\anaconda3\python.exe" `
  -NPaths 10000 `
  -Device auto `
  -FmNSteps 20 `
  -FmSolver euler `
  -Force
```

## 交付权重与结果

见 [`release/README.md`](release/README.md)。`release/checkpoints/` 当前包含 13 个一步学生 checkpoint（flow-map / CD / MeanFlow / on-policy × 7 / pricing × 3），`release/results/` 保留旧协议的原始评估 JSON。

下载到的权重包为 `checkpoints.tar.gz`。在仓库根目录解压即可恢复 `release/checkpoints/`：

```bash
# 1) 将 checkpoints.tar.gz 放到 release/ 下
# 2) 解压生成 release/checkpoints/...
tar -xzf release/checkpoints.tar.gz -C release/

# 3) 可选：校验 sha256
sha256sum -c release/checkpoints.tar.gz.sha256
```

注意：当前本地 release 中 `joint_fm_teacher` 只有配置 / 摘要，没有实际 teacher 权重 `ema_epoch_060.pt`；若要直接评估 teacher，需要重新训练或把 teacher checkpoint 放回对应目录。

## 配图复现

```bash
python3 analysis/make_figures.py
```

脚本读取 `analysis/viz_data/`，输出到 `analysis/figures/`。

## 测试

```bash
python3 -m pytest tests/
```

测试覆盖 Heston QE、区制模拟、MC oracle、期权定价、两阶段 / 联合 FM 训练器、MeanFlow / Consistency / flow-map 蒸馏、采样器、自回归 rollout、风格化事实、Wasserstein 与 signature 距离、Quant-GAN、on-policy 和 pricing 微调。
