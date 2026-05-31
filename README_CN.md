<p align="center">
<h1 align="center"><strong>Learning Soccer Skills for Humanoid Robots</strong></h1>
<h3 align="center">面向人形机器人的足球技能学习：渐进式感知-动作框架</h3>
<p align="center">
<a href="https://kongjipeng.github.io/" target="_blank">Jipeng Kong<sup>*</sup></a>,
<a href="https://xinzheliu.github.io/" target="_blank">Xinzhe Liu<sup>*</sup></a>,
Yuhang Lin,
<a href="https://bwrooney82.github.io/" target="_blank">Jinrui Han</a>,
<a href="https://sist.shanghaitech.edu.cn/soerensch_en/main.htm" target="_blank">Sören Schwertfeger<a>,
<a href="https://baichenjia.github.io/" target="_blank">Chenjia Bai<sup>&dagger;</sup></a>,
<a href="https://scholar.google.com.hk/citations?user=ahUibskAAAAJ" target="_blank">Xuelong Li<sup>&dagger;</sup></a>
<br>
<sup>*</sup> First Author &nbsp;&nbsp; <sup>&dagger;</sup> Corresponding Author
</p>
</p>

<div id="top" align="center">

[![Project](https://img.shields.io/badge/Project-Page-lightblue)](https://soccer-humanoid.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2602.05310-A42C25?style=flat&logo=arXiv&logoColor=A42C25)](https://arxiv.org/abs/2602.05310)
[![PDF](https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow)](https://soccer-humanoid.github.io/static/Soccer_arxiv.pdf)
[![Code](https://img.shields.io/badge/Code-GitHub-black?style=flat&logo=github)](https://github.com/TeleHuman/HumanoidSoccer)

</div>

---

## 项目概述

本仓库包含 **PAiD (Perception-Action integrated Decision-making)** 的官方实现，一个面向人形机器人足球技能学习的渐进式感知-动作框架。

框架分为三个阶段：
1. **动作技能习得** — 通过人体动作追踪学习全身运动
2. **感知-动作融合** — 轻量级的位置泛化能力
3. **物理仿真到真机迁移** — 基于物理的 Sim-to-Real 迁移

实验在 **Unitree G1** 机器人上验证，支持静态/滚动球、多位置、外部干扰、室内/室外等场景。

[![teaser](media/teaser.png "teaser")]()

---

## 代码结构

```
HumanoidSoccer/
├── source/whole_body_tracking/soccer/    # 核心任务/环境代码
│   └── tasks/tracking/
│       ├── mdp/
│       │   ├── rewards.py                # 踢球（Kicking）Reward
│       │   ├── rewards_dribbling.py      # 盘带（Dribbling）Reward
│       │   ├── observations.py           # 观测函数
│       │   ├── terminations.py           # 终止条件
│       │   ├── events.py                 # 事件/扰动函数
│       │   └── commands_multi_motion_soccer.py  # 动作指令系统
│       └── config/g1/
│           ├── soccer_flat_env_cfg.py    # 踢球环境配置
│           ├── soccer_dribbling_env_cfg.py  # 盘带环境配置
│           └── __init__.py               # Gym 环境注册
├── scripts/
│   ├── rsl_rl/                           # 训练 & 推理入口
│   ├── convert_gmr_to_soccer.py          # GMR 动作数据转换
│   ├── pkl_to_npz.py                     # PKL → NPZ 转换
│   └── replay_npz.py                     # 动作可视化
├── shell/
│   ├── progressive_soccer_train_play.sh  # 踢球渐进式训练脚本
│   └── progressive_dribbling_train.sh    # 盘带渐进式训练脚本
└── motions/                              # 动作数据集
    ├── soccer-standard/                  # 标准踢球动作
    └── hmr4d_4_unitree_g1_compatible.npz # GMR 转换的动作
```

---

## 安装

### 1. 安装 Isaac Lab v2.1.1

按照 [官方安装指南](https://isaac-sim.github.io/IsaacLab/v2.1.1/source/setup/installation/pip_installation.html) 安装，推荐使用 Pip 安装方式。

### 2. 克隆仓库

```bash
# SSH
git clone git@github.com:TeleHuman/HumanoidSoccer.git

# HTTPS
git clone https://github.com/TeleHuman/HumanoidSoccer.git
```

### 3. 安装依赖

```bash
pip install -e source/whole_body_tracking
```

### 4. 激活环境

```bash
conda activate HumanoidSoccer
```

---

## 已注册的 Gym 环境

| 环境 ID | 用途 |
|---------|------|
| `Tracking-Terrain-G1-RNN-v0` | Stage 1: 粗糙地形运动追踪 |
| `Tracking-Flat-G1-SoccerDestination-v0` | Stage 2: 踢球（MLP） |
| `Tracking-Flat-G1-SoccerDestination-RNN-v0` | Stage 2: 踢球（RNN） |
| `Tracking-Flat-G1-Dribbling-v0` | Stage 2: 盘带（MLP） |
| `Tracking-Flat-G1-Dribbling-RNN-v0` | Stage 2: 盘带（RNN） |
| `Tracking-Flat-G1-Dribbling-AnkleDisturb-v0` | Stage 1: 脚踝扰动盘带（MLP） |
| `Tracking-Flat-G1-Dribbling-AnkleDisturb-RNN-v0` | Stage 1: 脚踝扰动盘带（RNN） |

---

## 踢球任务（Kicking）

### 单阶段训练

```bash
python scripts/rsl_rl/train_multi.py --task Tracking-Flat-G1-SoccerDestination-RNN-v0 \
    --motion_path motions/soccer-standard \
    --num_envs 8192 \
    --headless
```

### 渐进式训练（2 阶段）

```bash
bash shell/progressive_soccer_train_play.sh test
```

| Stage | 环境 | 迭代数 | 说明 |
|-------|------|--------|------|
| 1 | `Tracking-Terrain-G1-RNN-v0` | 4000 | 粗糙地形上学走路 + 平衡 |
| 2 | `Tracking-Flat-G1-SoccerDestination-RNN-v0` | 默认 | 平地上学踢球 |

### 推理播放

```bash
python scripts/rsl_rl/play_multi.py --task Tracking-Flat-G1-SoccerDestination-RNN-v0 \
    --motion_path motions/soccer-standard \
    --num_envs 1
```

> ⚠️ **注意**：checkpoint 必须与环境匹配。MLP 训练的权重不能用 RNN 环境加载，反之亦然。


---

## Sprint 2 & 3: Anchor改动与状态机实验复盘

### 1. 当前最佳版本：连续动作极坐标观测 (Sprint 2)
在 **不切割运动序列**，保持原始 112 帧完整跑+踢（无缝播放）的前提下，我们做出了三点最关键的介入：
- **极坐标球位观测 (`observations_anchor.py`)**: 将踢球目标位置由原本的绝对坐标 XYZ 替换成了机器人的**自我中心极坐标**。
- **降权步态速度约束**: 因为 HMR 视觉动捕提取的速度包含极大噪声，我们大幅降低了速度追踪（`velocity tracking`）的权重。
- **剥离发力脚 (Ankle Masking)**: 从关节位置追踪的 Penalty 里去掉了右脚踝。让右腿在最后能根据球位自由发力挥扫，不受固定运动框架的拘束。
**这个连续跑踢的 Anchor Baseline 现阶段取得了最佳的视觉仿真连贯度，并成为了我们当前推进的主线 （`Anchor-Kick-G1-Soccer-RNN-v0` Task）。**

### 2. XGen "时空锚定" 状态机实验 (Sprint 3)
为了能够解耦奔跑跑动和最终一击的耦合度并允许球位高度自由，我们参考 XGen 将数据切分为了接触（`strike`）和非接触（`approach`）。
我们实现了 `commands_anchor.py` 并在 `sm_v6_xgen` 次方验证：
- 设计状态机，在靠近球 `$d \le 0.8$m` 时强制覆盖轨迹并应用坐标偏移（Correction XY）向真实物理球位强制“锚定坐标系”。
- **实验结果**: `kick_success_rate` 数值惊人地从 `0.45` 暴涨至 `0.70`。但在仿真渲染中，强制切换 motion index 与极强对齐修正带来的“硬切跳跃感”让机器人步态瞬间产生撕裂般的扭曲断层。
- **决定**: SM 的逻辑天花板极高，但平滑过渡机制还未成熟。现阶段代码已经作为 Sprint 3 遗产留在了我们的文件树中，主线目前**回退到 Sprint 2 连续不切断** 的基础。

#### 核心常用指令：继续训练目前最佳的连续动作版本

```bash
# 继续训练目前最佳的连续动作极坐标版本 (Task: Anchor-Kick-G1-Soccer-RNN-v0)
# (基于 run: 2026-04-10_09-47-29_anchor_resume)
CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/train_multi.py \
    --task Anchor-Kick-G1-Soccer-RNN-v0 \
    --motion_path motions/Video \
    --load_run "2026-04-10_09-47-29_anchor_resume" \
    --resume True --num_envs 2048 --device cuda:0 --headless



CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/train_multi.py \
    --task Anchor-CG-Kick-G1-Soccer-RNN-v0 \
    --motion_path motions/Video \
    --load_run "2026-04-10_02-11-14" \
    --checkpoint model_6999.pt \
    --run_name cg_v73_phase_split \
    --resume True \
    --num_envs 2048 --device cuda:0 --headless --max_iterations 20000

# 播放上面训练版本的动作推演
CUDA_VISIBLE_DEVICES=1 python scripts/rsl_rl/play_multi.py \
    --task Anchor-Kick-G1-Soccer-RNN-v0 \
    --motion_path motions/Video \
    --load_run "2026-04-10_09-47-29_anchor_resume" \
    --num_envs 1 --device cuda:0
```

---

## Sprint 4: 软性接触图 (Soft Contact Graph)

### 核心思想
在不硬切 motion 的前提下，利用 npz 中已标注的 `kick_frame` / `kick_end_frame` 将每个 episode 分为两个时间阶段：

| 阶段 | 帧范围 | 含义 |
|------|--------|------|
| **CG=0** (Approach) | `t < kick_frame - 5` | 禁止碰球，右脚踝正常追踪 |
| **CG=1** (Kick) | `t ≥ kick_frame - 5` | 鼓励踢球，右脚踝追踪降权至 `0.3×`（soft guidance） |

> ⚠️ CG 只有 0 和 1 两个阶段。`kick_end_frame` 仅用于 `interaction_termination`（防摆烂），不构成独立阶段。

### 新增 Reward / Termination

| 组件 | 文件 | 效果 |
|------|------|------|
| `early_collision_penalty` | `rewards.py` | CG=0 碰球 → 持续惩罚（weight: -10.0） |
| `time_gated_contact` | `rewards.py` | 替代原 `target_point_contact`，CG=0 碰球不给奖 |
| `dynamic_ankle_masking_body_pos` | `rewards.py` | CG=0 全身追踪；CG=1 踢球脚误差 ×0.3（保留踢球姿态） |
| `interaction_termination` | `terminations.py` | `kick_end_frame` 后球速≈0 → 终止（防摆烂） |

### 设计决策

- **Soft vs. Hard CG**：采用 reward shaping（soft）而非状态机硬切换，保持连续动作的稳定性
- **Soft Guidance（×0.3）**：CG=1 阶段不完全放开脚踝追踪，而是降权到 0.3×，防止 agent 用脚底拉球而非脚背抽球
- **球位前移**：`radius` 范围调整为 `0.0~0.4`，减少 approach 阶段的意外碰撞

### 常用指令

```bash
# Sprint 4 训练（从 Stage 1 checkpoint 开始）
CUDA_VISIBLE_DEVICES=1 python scripts/rsl_rl/train_multi.py \
    --task Anchor-CG-Kick-G1-Soccer-RNN-v0 \
    --motion_path motions/Video \
    --load_run "2026-04-10_02-11-14" --checkpoint model_6999.pt \
    --run_name "cg_v3_softmask" --resume True \
    --num_envs 2048 --device cuda:0 --headless

# 播放推演（带显示器）
CUDA_VISIBLE_DEVICES=1 python scripts/rsl_rl/play_multi.py \
    --task Anchor-CG-Kick-G1-Soccer-RNN-v0 \
    --motion_path motions/Video \
    --load_run "2026-04-28_12-15-12_cg_v3_softmask" \
    --checkpoint model_12000.pt \
    --num_envs 1 --device cuda:0
```

---

## Headless 视频录制（SSH 远程适用）

### 单视角录制

在没有显示器的 SSH 环境下录制 demo 视频：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/play_multi.py \
    --task Anchor-CG-Kick-G1-Soccer-RNN-v0 \
    --motion_path motions/Video \
    --load_run "<your_run_name>" \
    --checkpoint model_XXXX.pt \
    --num_envs 1 --device cuda:0 \
    --headless --video --video_length 600
```

- `--video`: 自动启用 offscreen 渲染摄像头
- `--video_length N`: 录制 N 步（默认 600）
- 视频保存在 `logs/.../videos/play_<timestamp>/`

### 双视角分屏录制（带 CG HUD）

录制正面+背面分屏视频，画面上叠加 Contact Graph 阶段、帧计数和球速：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/play_multi.py \
    --task Anchor-CG-Kick-G1-Soccer-RNN-v0 \
    --motion_path motions/Video \
    --load_run "2026-04-28_12-15-12_cg_v3_softmask" \
    --checkpoint model_12000.pt \
    --num_envs 1 --device cuda:0 \
    --headless --dual_view --video_length 1000 --path_tracing
```

- `--dual_view`: 前后双摄像头分屏，自动跟随机器人
- `--path_tracing`: （可选）切换到 Path Tracing 渲染器，画质更好但速度慢
- 视频 HUD 显示：`Step / Frame`, `CG Phase`, `Ball Speed (m/s)`
- 输出：`logs/.../videos/dual_<timestamp>/dual_view_*.mp4`

> ⚠️ 需系统安装 `ffmpeg`。双视角分辨率为 1920×540（左右各 960×540）。








CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/eval_kick_diagnostic.py     --task Anchor-CG-Kick-G1-Soccer-RNN-v0     --motion_path motions/Video     --load_run "2026-05-06_23-52-33_cg_v74_sparse_bonus"     --checkpoint model_26000.pt     --num_envs 16 --eval_episodes 50 --headless

---

## 盘带任务（Dribbling）

### 任务定义

机器人的唯一目标：**带着球跑，不能把球弄丢**。

### 盘带专属 Reward

| Reward | 权重 | 逻辑 |
|--------|------|------|
| **速度一致性** (Velocity Tracking) | +5.0 | `exp(-‖v_ball − v_pelvis‖²/σ²)` — 球速必须与骨盆速度对齐 |
| **动态距离** (Dynamic Proximity) | +5.0 | 球在机器人本地坐标系前方 [0.2m, 0.5m] 安全区才满分 |
| **微接触过滤** (Micro-Contact Filter) | −10.0 | 5 帧 EMA 低通滤波，接触力超 20N 才罚分，单步上限 2.0 |

### 终止条件

- `ball_lost`：球-骨盆距离 > 1.0m 或速度差 > 2.0 m/s 时，立即终止回合（前 50 步 grace period）

### 渐进式训练

```bash
# 默认模式（Stage 1 使用 vanilla terrain）
bash shell/progressive_dribbling_train.sh my_run

# 脚踝扰动模式（Stage 1 零化脚踝追踪 + 随机力矩扰动）
bash shell/progressive_dribbling_train.sh my_run --ankle-disturb
```

| Stage | 默认环境 | 脚踝扰动环境 | 说明 |
|-------|---------|------------|------|
| 1 | `Tracking-Terrain-G1-RNN-v0` | `Tracking-Flat-G1-Dribbling-AnkleDisturb-RNN-v0` | 运动基础 |
| 2 | `Tracking-Flat-G1-Dribbling-RNN-v0` | 同左 | 盘带技能 |

日志目录：`logs/rsl_rl/g1_dribbling/`（与踢球的 `g1_flat/` 完全隔离）

### 脚踝扰动模式

在 Stage 1 训练时：
- **`motion_foot_pos` 权重 = 0** — 不追踪脚踝轨迹（视频数据精度差）
- **每 0.1-0.3s 注入 ±15 N·m 随机力矩** — 四个脚踝关节（左右 pitch/roll）
- **效果**：机器人被迫用躯干核心 + 支撑腿维持平衡

### 单独测试盘带环境

```bash
python scripts/rsl_rl/train_multi.py --task Tracking-Flat-G1-Dribbling-v0 \
    --motion_path motions/soccer-standard \
    --num_envs 16 --headless --max_iterations 5
```

### 推理播放

```bash
python scripts/rsl_rl/play_multi.py --task Tracking-Flat-G1-Dribbling-RNN-v0 \
    --motion_path motions/soccer-standard \
    --num_envs 1
```

---

## 动作数据

### 可视化动作文件

```bash
# 标准踢球动作
python scripts/replay_npz.py --motion_path motions/soccer-standard/soccer-standard-002_right.npz

# GMR 转换的动作
python scripts/replay_npz.py --motion_path motions/Video/hmr4d_1_unitree_g1_compatible.npz
```

### 从 GMR 转换动作数据

```bash
# 1. 使用 GMR 导出 .pkl 文件

# 2. 转换为 HumanoidSoccer 兼容格式（带朝向归一化）
#    --normalize_yaw 将初始朝向旋转到 -90°（面向 -Y），与 MoCap 数据对齐
python scripts/convert_gmr_to_soccer.py \
    --input motions/pkl/hmr4d_1_unitree_g1.pkl \
    --output motions/pkl/hmr4d_1_unitree_g1_compatible.pkl \
    --normalize_yaw

# 3. 转换为 .npz（需要 Isaac Sim 环境）
#    --kick_leg 指定踢球脚（left 或 right），会写入 npz 的 kick_leg 字段
#    --output_name 指定输出文件名（不含 .npz 后缀）
python scripts/pkl_to_npz.py \
    --input_file motions/pkl/hmr4d_1_unitree_g1_compatible.pkl \
    --output_name hmr4d_1_unitree_g1_compatible_right \
    --output_dir motions/Video \
    --kick_leg right \
    --headless

# 4.（可选）为已有 .npz 文件添加 kick_leg 标签
python scripts/kick_motion_label.py motions/Video/hmr4d_1_unitree_g1_compatible.npz --label right
```

> ⚠️ GMR 导出的数据已经是 XYZW 四元数格式，不需要额外的坐标转换。
> 
> ⚠️ **必须指定 `--kick_leg`**，否则训练时 `MultiMotionLoader` 无法区分左右脚，会导致踢球脚判断错误。
>
> ⚠️ **建议始终使用 `--normalize_yaw`**，确保 HMR 动作的初始朝向与 MoCap 标准数据（面向 -Y 方向）一致，否则球放置位置和踢球方向会出错。

---

## 训练监控（TensorBoard）

训练过程中可以用 TensorBoard 实时查看 reward 曲线、损失函数和终止统计：

```bash
# 监控踢球训练
tensorboard --logdir logs/rsl_rl/g1_flat --port 6006

# 监控盘带训练
tensorboard --logdir logs/rsl_rl/g1_dribbling --port 6006

# 同时监控所有实验
tensorboard --logdir logs/rsl_rl --port 6006
```

然后在浏览器中打开 `http://localhost:6006`（如果是远程服务器，需要 SSH 端口转发）：

```bash
# SSH 端口转发（本地机器执行）
ssh -L 6006:localhost:6006 user@remote-server
```

### 关键指标

| 指标 | 含义 | 期望趋势 |
|------|------|---------|
| `Reward/dribbling_velocity_tracking` | 球-骨盆速度一致性 | ↑ 上升 |
| `Reward/dribbling_dynamic_proximity` | 球在安全区比例 | ↑ 上升 |
| `Reward/dribbling_legal_foot_touch` | 合法脚轻触次数 | ↑ 上升 |
| `Reward/dribbling_micro_contact_filter` | 合法脚重击惩罚 | ↓ 下降 |
| `Reward/dribbling_undesired_contact_penalty` | 非法触球惩罚 | ↓ 趋近 0 |
| `Episode_Termination/ball_lost` | 丢球终止比例 | ↓ 下降 |

---

## TODO

- [x] 发布 PAiD 训练代码
- [x] 发布 PAiD 动作数据集
- [x] 实现盘带（Dribbling）任务环境
- [ ] 发布 PAiD Domain Randomization 代码

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{kong2026learningsoccerskillshumanoid,
  title={Learning Soccer Skills for Humanoid Robots: A Progressive Perception-Action Framework},
  author={Jipeng Kong and Xinzhe Liu and Yuhang Lin and Jinrui Han and Sören Schwertfeger and Chenjia Bai and Xuelong Li},
  year={2026},
  eprint={2602.05310},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2602.05310}
}
```

## 许可证

本代码库采用 [CC BY-NC 4.0 许可证](https://creativecommons.org/licenses/by-nc/4.0/deed.zh-hans)。不得将本项目用于商业用途。

## 联系方式

如有合作需求或问题讨论，请联系：

- 第一作者：Jipeng Kong [kongjp2024@shanghaitech.edu.cn](mailto:kongjp2024@shanghaitech.edu.cn)，Xinzhe Liu [liuxzh2023@shanghaitech.edu.cn](mailto:liuxzh2023@shanghaitech.edu.cn)
- 通讯作者：Chenjia Bai [baicj@chinatelecom.cn](mailto:baicj@chinatelecom.cn)
