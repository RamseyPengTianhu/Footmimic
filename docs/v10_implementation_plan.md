# v10 Target-Conditioned Adaptive Kick — Implementation Plan

## 总体评价

这是目前为止最合理的下一步方案。原因：

1. **基于实验证据驱动** — v3 grid 证明 ±10cm 静态泛化不是瓶颈；v9a 证明盲目加球速会导致 timing 崩溃（CΔ=+62）
2. **不破坏已有能力** — partial loading + time-aligned curriculum 保证 v10 初始行为 ≈ v3
3. **不重复失败实验** — 不加新 reward（v71~v8a 全部失败），只改 obs + task input
4. **渐进式难度** — D0→D5 curriculum 每步只引入一个变化

---

## Step 1 分析：Time-Aligned Rolling Ball

### 核心公式

```
t_arrive = nominal_kick_frame × dt
ball_init_xy = nominal_contact_xy − ball_vel_xy × t_arrive
```

> [!IMPORTANT]
> 这是整个方案最关键的设计。它保证了 v3 的核心假设："球在该被踢的时候，在该在的位置"。

### 实现细节

需要在 `_compute_soccer_ball_positions()` 中：
1. 计算 `nominal_contact_xy`（当前的 `target_xy`，即 motion 轨迹终点）
2. 采样 `ball_vel_xy`（方向 + 速度）
3. 反推 `ball_init_xy = nominal_contact_xy - ball_vel_xy * t_arrive`
4. 将 `ball_init_xy` 设为球初始位置，`ball_vel_xy` 设为球初始速度

### 潜在问题

**Approach 阶段的 observation 漂移**：

虽然球在 `kick_frame` 到达正确位置，但在 approach 阶段（t=0 到 t=kick_frame），球在路上。`anchor_ball_polar` 会报告与 v3 训练时完全不同的值：

```
v3 训练时：approach 阶段 ball_polar ≈ 固定值（球不动）
v10 训练时：approach 阶段 ball_polar 持续变化（球在移动）
```

这意味着：
- v3 的 LSTM hidden state 在 approach 阶段会被"不同的 ball_polar 序列"干扰
- 但因为 partial loading 把新 obs weights 初始化为 0，`ball_vel` 等新 obs 一开始不影响 action
- **RNN 需要重新学习 approach 阶段的 ball_polar 变化 pattern** — 这是 fine-tune 的核心任务

**建议**：D0 → D1 之间可能需要比预想的更多 iterations（5k-10k 而非 3k-5k）

### 球方向选择

从哪个方向滚过来？建议：

- **D1 初始**：从 approach 方向的反方向滚过来（球从前方滚向 contact point），最符合"传球后截击"场景
- **D1 后期**：加入 ±30° 方向随机，模拟斜传
- **D3+**：加入 ±90° 方向随机

---

## Step 2 分析：Ball-State / Intercept Observations

### 推荐 obs（按重要性排序）

| Obs | Dim | 计算方式 | 重要性 |
|-----|:---:|---------|:------:|
| `ball_vel_local` | 3D | 球速投影到 pelvis frame | ★★★ |
| `time_to_contact` | 1D | `(desired_contact_frame - t) × dt` | ★★★ |
| `intercept_point_local` | 3D | predicted ball pos at desired_contact_frame, in pelvis frame | ★★★ |
| `ball_rel_swing_foot` | 3D | ball_pos - swing_foot_pos in pelvis frame | ★★ |
| `ball_rel_support_foot` | 3D | ball_pos - support_foot_pos in pelvis frame | ★★ |
| `kick_side` | 1D | +1 right, -1 left | ★ |
| `kick_phase` | 1D | normalized t/T | ★ |
| `target_dir_local` | 2D | [cos, sin] of kick direction in pelvis frame | ★ |

**Total: +18D**（160D → 178D）

### 关键语义问题

**`intercept_point` 怎么算？**

- **D0-D1（time-aligned）**：`intercept_point = nominal_contact_pos`（trivially 等于球的目标位置）
- **D2+（timing jitter）**：`intercept_point = ball_pos + ball_vel × time_to_contact`（线性外推）
- **D4+（position jitter）**：同上，但考虑 jitter 后的实际球轨迹

**`time_to_contact` 语义**：
- 等于 `(desired_contact_frame - current_frame) × dt`
- 当 `current_frame > desired_contact_frame` 时为负值 → policy 知道"球已经过了预期接触点"
- **不需要 learned prediction** — 直接用 desired_contact_frame（D0-D1 就是 nominal_kick_frame）

### 加在 actor 还是 critic？

**全部加在 actor**。原因：
- Policy 需要这些信息做决策（什么时候踢、怎么调整步伐）
- Critic 已经有 292D privileged obs，足够

---

## Step 3 分析：Partial Loading

### LSTM Partial Loading 实现

```python
# Actor RNN: LSTM(input_size=160 → 178, hidden_size=128, num_layers=2)

old_state_dict = torch.load("model_12000.pt")

# For each LSTM layer:
for layer_idx in [0, 1]:
    key_ih = f"rnn.weight_ih_l{layer_idx}"
    key_hh = f"rnn.weight_hh_l{layer_idx}"
    key_bias_ih = f"rnn.bias_ih_l{layer_idx}"
    
    old_ih = old_state_dict[f"actor_rnn.{key_ih}"]  # [4*128, old_input]
    new_ih = torch.zeros(4*128, new_input_size)
    new_ih[:, :old_input_size] = old_ih  # 新 obs 列 = 0
    
    # weight_hh 不变（hidden→hidden）
    # bias 不变

# For first MLP layer after RNN (if any):
# Same pattern: zero-pad columns for new input dimensions
```

> [!WARNING]
> RSL_RL 的 `ActorCriticRecurrent` 结构：input → RNN → MLP。只有 RNN 的 `weight_ih` 维度变化（第一层），MLP 的 input 来自 RNN hidden（128D，不变）。所以**只需要改 LSTM layer 0 的 weight_ih**。

### 实际实现位置

需要在 `train_multi.py` 的 `runner.load()` 之前/之后加 partial loading hook，或者写一个独立的 checkpoint conversion script。

---

## Step 4 分析：Same Reward

完全同意。v71~v8a 的实验已经证明：

- Support foot reward → 与 tracking 冲突
- Orientation reward → 与 pelvis_orientation 冲突  
- Post-strike stability → PPO credit leakage
- Ankle lock → 无效

**唯一需要注意的**：`early_collision_penalty` 的时序判定。

当前 early collision 判定基于 `kick_frame`。D2+ 阶段如果 `desired_contact_frame` 变了，early collision 的判定也需要跟着变。但这属于 Step 5 的工作，v10a 暂时不需要改。

---

## Step 5 分析：Dynamic Contact Frame

这是从 static kick → adaptive kick 的关键架构转变。

**当前**：
```
valid_contact_window = [kick_frame, kick_end_frame]  # 固定标注
early_collision = contact_before(kick_frame - margin)
```

**v10c/d 目标**：
```
desired_contact_frame = compute_intercept_time(ball_pos, ball_vel)
valid_contact_window = [desired_contact_frame - 5, desired_contact_frame + 5]
early_collision = contact_before(desired_contact_frame - margin)
```

> [!IMPORTANT]
> 这步暂时不做（v10a 仍用 nominal_kick_frame）。但需要提前设计好 `desired_contact_frame` 的接口，方便后续替换。

---

## Step 6 分析：Phase-Specific Tracking Relaxation

同意延后。只有以下条件同时满足时才考虑：

1. v10a/b 有 ball_vel / intercept obs ✓
2. Policy 仍不调整 timing ✗
3. CΔ 仍然很大 ✗

如果需要做，建议方式：

```python
# Phase-gated tracking weight
tracking_scale = torch.ones(num_envs)
tracking_scale[strike_mask] = 0.3  # 降低 strike 阶段的 tracking
# 只降 body_pos / body_ori，不降 foot_pos（foot_pos 在 strike 阶段仍然重要）
```

---

## Curriculum 建议

| Phase | 球状态 | 新 Obs | Arrive Time | Position | Speed | 预计 Iterations |
|-------|--------|--------|:-----------:|:--------:|:-----:|:---------------:|
| D0 | 静态 | +18D (zero-init) | exact | exact | 0 | 3k（验证 partial load 不崩） |
| D1 | 滚动 | 同上 | exact | exact | 0.05-0.15 m/s | 5k |
| D2 | 滚动 | 同上 | ±5 frames | exact | 0.05-0.15 m/s | 3k |
| D3 | 滚动 | 同上 | ±10 frames | exact | 0.10-0.20 m/s | 3k |
| D4 | 滚动 | 同上 | ±10 frames | ±5cm | 0.10-0.20 m/s | 3k |
| D5 | 滚动 | 同上 | ±10 frames | ±10cm | 0.20-0.30 m/s | 5k |

**总计约 22k iterations**

### 通过标准

| Phase | Kick% | BSpd | CΔ | EarlyCol | Fall |
|-------|:-----:|:----:|:--:|:--------:|:----:|
| D0 | ≥95% | ≥7.5 | ≤5 | 0% | ≤5% |
| D1 | ≥90% | ≥7.0 | ≤10 | ≤2% | ≤8% |
| D2 | ≥85% | ≥6.5 | ≤15 | ≤5% | ≤10% |
| D5 | ≥80% | ≥6.0 | ≤10 | ≤5% | ≤12% |

---

## 具体文件改动

### 新增文件

#### [NEW] `scripts/rsl_rl/convert_checkpoint_v10.py`
Partial loading script：读取 v3 checkpoint，扩展 LSTM weight_ih，zero-pad 新 obs 列，输出 v10 initial checkpoint。

### 修改文件

#### [MODIFY] `commands_multi_motion_soccer.py`
- `_compute_soccer_ball_positions()`：加入 time-aligned rolling ball 逻辑
- `_update_soccer_ball()`：设置反推的 ball_init_pos + ball_vel
- 新增 `desired_contact_frame` 属性（暂时 = nominal_kick_frame）

#### [MODIFY] `observations_anchor.py`
- 新增 `ball_vel_local()`: 球速投影到 pelvis frame
- 新增 `time_to_contact()`: (desired_contact_frame - t) × dt
- 新增 `intercept_point_local()`: predicted ball pos at contact
- 已有 `ball_relative_feet()`, `kick_context()`, `target_direction_local()` 可复用

#### [MODIFY] `soccer_anchor_env_cfg.py`
- 新增 `G1AnchorTargetConditionedKickEnvCfg`（v10）
- 加入所有新 obs terms
- 加入 ball curriculum config

#### [MODIFY] `__init__.py`
- 注册 `Anchor-Target-Kick-G1-Soccer-RNN-v0`

### 不改的文件

- `rewards.py` — 不加新 reward
- `terminations.py` — 暂不改 early_collision 判定
- `train_multi.py` — 加 partial loading 选项

---

## 验证计划

### D0 验证
```bash
# 1. 转换 checkpoint
python scripts/rsl_rl/convert_checkpoint_v10.py \
    --input logs/.../model_12000.pt \
    --output logs/.../model_12000_v10_init.pt \
    --old_obs_dim 160 --new_obs_dim 178

# 2. 用 v10 config + 静态球 eval（验证 partial load 不崩）
python scripts/rsl_rl/eval_kick_diagnostic.py \
    --task Anchor-Target-Kick-G1-Soccer-RNN-v0 \
    --checkpoint model_12000_v10_init.pt \
    --eval_episodes 20

# 3. 预期：Kick%≈97%, BSpd≈8.0（≈ v3 baseline）
```

### D1 验证
```bash
# Time-aligned rolling ball eval
# 预期：Kick%≥90%, CΔ≤10, BSpd≥7.0
```

---

## Open Questions

> [!IMPORTANT]
> **Q1: ball_vel 方向的采样策略**
> 
> D1 阶段球从哪个方向滚过来？建议从 approach 反方向（球从前方滚来），但也可以随机方向。后者更难但泛化更强。你倾向哪个？

> [!IMPORTANT]
> **Q2: partial loading 的实现位置**
> 
> 选项 A：独立 script `convert_checkpoint_v10.py`，生成新 checkpoint 文件
> 选项 B：在 `train_multi.py` 的 `runner.load()` 里加 hook
> 
> 选项 A 更干净（不改训练代码），但需要手动跑一步。你选哪个？

> [!IMPORTANT]
> **Q3: Curriculum 切换方式**
> 
> 选项 A：手动切换（每个 phase 跑完后改 config 再 resume）
> 选项 B：自动 curriculum（根据 Kick% 自动升级难度）
> 
> 选项 A 更可控但费人工。选项 B 更优雅但需要额外实现。
