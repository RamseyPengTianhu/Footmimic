# v10 Phase-Retiming Adapter — Analysis

## 1. 为什么这比之前的 v10 方案更好

| | 之前的 v10 (flat obs expansion) | 新的 v10 (phase adapter) |
|---|---|---|
| v3 状态 | **被修改**（partial loading, fine-tune） | **完全冻结** |
| 失败风险 | 可能破坏 v3 的 kick 能力 | 零风险（v3 不变） |
| 学什么 | 整个 29D action 空间 | **只学 1D Δτ** |
| 对准的问题 | "给 policy 更多信息让它自己想办法" | **直接解决 CΔ=+62 的 timing 问题** |
| 可解释性 | 低（29D action 变化难分析） | **高**（Δτ 直接可读：提前/延后多少帧） |
| 扩展性 | 需要重训整个 policy | 只扩展 HL 输出维度 |

**核心洞察**：v3 的问题不是"不会踢球"，而是"踢球时机被 reference phase 锁死"。Phase adapter 只解锁 timing，不动 skill。

---

## 2. 架构分析

```
┌─────────────────────────────────────────────┐
│  High-Level Phase Adapter (可训练)            │
│                                              │
│  obs_hl ──► MLP(64,32) ──► Δτ (1D)          │
│  ~30D input              clamped [-K, K]     │
└──────────────────┬──────────────────────────┘
                   │ Δτ
                   ▼
         τ = clamp(t + Δτ, 0, T-1)
                   │
                   ▼
        ref_τ = motion_lib[τ]    ← motion query 用 τ
                   │
┌──────────────────┴──────────────────────────┐
│  v3 Low-Level Policy (冻结)                  │
│                                              │
│  (obs_t, ref_τ) ──► LSTM ──► 29D action     │
│  160D input                                  │
└─────────────────────────────────────────────┘
```

### 关键点

- v3 的 LSTM 接收的 `command` obs 维度不变（58D），但其内容变了（来自 `motion[τ]` 而非 `motion[t]`）
- v3 不需要知道 Δτ 的存在——它只是照常做 motion tracking
- HL 只需要学会：球早到 → Δτ > 0（加快 phase）；球晚到 → Δτ < 0（延后 phase）

---

## 3. HL Input Design（~30D）

| Obs | Dim | 说明 |
|-----|:---:|------|
| `ball_pos_local_history` | 15D | 最近 5 帧球位 (pelvis frame)，隐式推断速度 |
| `ball_rel_swing_foot` | 3D | 球 - 踢球脚位置 |
| `ball_rel_support_foot` | 3D | 球 - 支撑脚位置 |
| `current_phase` | 1D | t / T |
| `kick_frame_countdown` | 1D | (kick_frame - t) × dt，负值表示过了 |
| `previous_delta_tau` | 1D | 上一步的 Δτ |
| `base_proj_gravity` | 3D | 平衡状态 |
| `base_ang_vel` | 3D | 角速度 |
| **Total** | **~30D** | |

> [!TIP]
> **不给显式 ball_vel 而是给 position history** 是正确选择：
> - 让 HL 自己学时序特征，不依赖外部估计器
> - 5 帧 history 对应 0.1s（50Hz），足够推断 0.05~0.3 m/s 的球速
> - 如果后续需要更长时序，可以扩到 10 帧或用 1D Conv

---

## 4. tau-space Reward Conversion（最难的部分）

### 4.1 当前的 time-dependent 逻辑

```python
# 以下都绑定 wall-clock t 和 fixed kick_frame
early_collision:     contact && t < kick_frame - margin
target_point_contact: contact && kick_frame <= t <= kick_end_frame
post_strike:         t > kick_end_frame
```

### 4.2 改成 tau-space

```python
tau = t + delta_tau  # HL 输出

# Phase masks 全部用 tau
strike_mask     = abs(tau - kick_frame) < strike_window
pre_strike_mask = tau < kick_frame
post_strike_mask = tau > kick_end_frame

# Contact reward 时序判定用 tau
early_collision:     contact && tau < kick_frame - margin
target_point_contact: contact && kick_frame <= tau <= kick_end_frame
```

### 4.3 语义一致性

这样改的含义是：

| 场景 | Δτ | 效果 |
|------|:--:|------|
| 球早到 5 帧 | +5 | tau 提前到达 kick_frame → contact window 提前打开 → 不判 early collision |
| 球晚到 5 帧 | -5 | tau 延后到达 kick_frame → policy 在 approach 阶段多停留 → 等球到了再踢 |
| 球准时 | 0 | tau = t → 退化为 v3 行为 |

> [!WARNING]
> **Tracking reward 的 reference state 也必须用 τ**
> 
> `motion_body_pos` 等 tracking reward 比较 `robot_state` vs `motion[τ]`。如果 tracking 用 `motion[t]` 但 command obs 用 `motion[τ]`，policy 会被矛盾信号撕裂。
> 
> 实现方式：在 `_update_command()` 里把 `time_steps` 替换为 `tau` 来查询 motion state。这意味着 `motion.get_motion_state()` 的输入从 `t` 变为 `τ`。

---

## 5. Δτ Regularization

需要一个小的正则项防止 Δτ 乱跳：

```python
delta_tau_reg = -0.01 * delta_tau**2  # L2 惩罚大偏移
delta_tau_rate = -0.005 * (delta_tau - prev_delta_tau)**2  # 平滑
```

权重要小——目的是让 Δτ 在没有球速信号时趋向 0（退化为 v3），而不是限制 adaptation 范围。

---

## 6. 实现复杂度评估

### 需要改的核心逻辑

| 改动 | 难度 | 文件 |
|------|:----:|------|
| HL policy network（MLP, ~30D→1D） | 低 | 新增 `phase_adapter.py` |
| motion query 用 τ 而非 t | **中** | `commands_multi_motion_soccer.py` |
| tau-space reward masks | **中** | `rewards.py`, `kick_detection.py` |
| HL obs functions | 低 | `observations_anchor.py` |
| Time-aligned rolling ball | 中 | `commands_multi_motion_soccer.py` |
| HL 训练循环（v3 frozen + HL trainable） | **高** | `train_multi.py` 或新脚本 |

### HL 训练循环的实现

最大挑战：RSL_RL 的 `ActorCriticRecurrent` 假设一个 policy。要做 hierarchical：

**选项 A：两阶段 forward pass**
1. HL policy forward → Δτ
2. 用 Δτ 修改 command obs（查 motion[τ]）
3. v3 frozen forward → action
4. HL 通过 PPO 更新，v3 不更新

实现：在 env 的 observation manager 里加一个 "pre-step hook"，让 HL 先算 Δτ，再修改 command。

**选项 B：把 HL 嵌入 env**
- HL 作为 env 的一部分（不是 policy）
- HL 在 `_update_command()` 里运行
- HL 的 action（Δτ）通过 env 的 action space 暴露
- v3 的 action space 变成 29D + 1D = 30D

这样 RSL_RL 看到的是一个 30D action 的 policy，其中前 29D 从 v3 frozen weights 出来，最后 1D 从 HL 出来。

> [!IMPORTANT]
> **选项 B 更实用但更 hacky**。选项 A 更干净但需要改训练循环。建议先用选项 B 快速验证，后续重构到选项 A。
> 
> 选项 B 实现：把 HL 的 Δτ 作为 action 的第 30 维，freeze v3 的 29D action head 的权重。但 RSL_RL 的 freeze 粒度不确定是否支持...
>
> 另一个选项 C：**不 freeze v3，但 Δτ regularization 保护**。让整个 30D policy 一起训练，但 v3 的 29D action head 从 v3 checkpoint 初始化，learning rate 极低（或前 N 步 frozen）。风险比完全 freeze 高，但实现简单得多。

---

## 7. 对 Roadmap 的评价

```
v10_phase_adapter → v10_structured_adapter → v11_video → v12_latent → v13_XKickGen
```

**评价：节奏合理**

- v10 phase adapter 是最小可行验证：能不能通过 1D Δτ 解决 timing？
- v10d structured residual 是自然扩展：timing 对了但 geometry 还差
- v11 video motion library 解决数据瓶颈（目前只有 2 条 motion）
- v12 latent 是架构升级（从 explicit phase 到 learned latent）
- v13 是终极目标（interaction synthesis）

每一步都有明确的 pass/fail 标准，不会浪费算力在错误方向。

---

## 8. 建议的执行顺序

### Phase 1: 基础设施（~2天）

1. 实现 HL obs functions（ball_pos_history, ball_rel_feet, countdown）
2. 实现 time-aligned rolling ball（反推 ball_init_pos）
3. 实现 tau-space motion query + reward mask conversion
4. 决定 HL 训练方式（选项 A/B/C）

### Phase 2: v10a Static Sanity（~1天训练）

- v3 frozen + HL zero-init
- 静态球
- 验证 Kick%≥95%, Δτ≈0

### Phase 3: v10b Slow Rolling（~2天训练）

- 0.03-0.05 m/s time-aligned
- 验证 Δτ 开始非零，CΔ→0

### Phase 4: v10c Timing Jitter（~3天训练）

- ±3 → ±5 → ±10 → ±20 frames
- 核心验证：ball early → Δτ>0, ball late → Δτ<0

---

## Open Questions

> [!IMPORTANT]
> **Q1: HL 训练方式**
>
> - 选项 A：clean hierarchical（改训练循环）
> - 选项 B：30D action，freeze 前 29D
> - 选项 C：30D action，v3 init + 低 lr，不 freeze
>
> 选项 C 最快能跑起来，但风险是 v3 能力退化。你怎么选？

> [!IMPORTANT]
> **Q2: Δτ 的执行频率**
>
> - 每步输出（50Hz）→ 最灵活但可能抖动
> - 每 5 步输出一次 → 更平滑但响应慢
> - 建议起步每步输出 + rate penalty

> [!IMPORTANT]
> **Q3: motion interpolation**
>
> τ = t + Δτ 可能是非整数。当前 `motion_lib.get_motion_state()` 是否支持 float-index 插值？如果不支持需要先加。
