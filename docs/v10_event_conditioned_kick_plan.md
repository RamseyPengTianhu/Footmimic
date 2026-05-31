# v10: Event-Conditioned Kick Decoder (Revised)

## Problem Statement

v3 学到的是 **"第 t 帧做什么"**，而不是 **"什么条件下该踢球"**。
- v3 不是 conditional tracker，而是 static kick executor
- Phase adapter 不可行：LSTM 内部状态和固定 reference timing 强绑定
- 辅助 reward 全部与 tracking reward 冲突
- 动态球实验（v9a）失败：timing 被固定 reference 锁死

**核心转变**：从 trajectory tracking → event-conditioned interaction skill

v3 的价值：初始化 checkpoint / teacher / static primitive baseline，不是最终可控底层。

---

## Architecture: MLP + Flattened History

不用 LSTM。用显式 history buffer + MLP，更简单、更好调试、BC 更容易。

```
┌───────────────────────────────────────────────────────────┐
│                    INPUTS (~420D)                         │
├────────────────┬──────────────────┬───────────────────────┤
│ Current ~64D   │ History ~290D    │ Task Condition ~66D   │
│ • ang_vel 3    │ • joint_pos ×3f  │ • event_obs 8D        │
│ • gravity 3    │   = 87D          │ • ball_foot_rel 24D   │
│ • joint_pos 29 │ • joint_vel ×3f  │ • weak_prior 8D       │
│ • joint_vel 29 │   = 87D          │ • ball_pos_hist ×10f  │
│                │ • action ×3f     │   = 30D (local XYZ)   │
│                │   = 87D          │                       │
│                │ • last_action 29 │                       │
├────────────────┴──────────────────┴───────────────────────┤
│              MLP: 512 → 256 → 128 → 29D action           │
└───────────────────────────────────────────────────────────┘
```

> [!NOTE]
> 50Hz 下 10 frames = 0.2s，对 0.03~0.6 m/s 球速足够估计运动趋势。
> 不需要整段 episode history——v10 是 event-conditioned control，不是 sequence modeling。

---

## 1. Event-Warped Weak Prior (核心设计)

### 问题

如果 retiming 改了 event boundary 但 motion reference 仍然按 `motion[t]` 播放，会造成新矛盾：

```
原始 kick_frame = 80, retiming 后 strike event = 65

t=65 时:
  event condition → "现在是 strike"
  contact graph → "现在应该击球"
  ball relation → "swing foot 应该接近球"
  weak prior → motion[65] = 原始 pre-strike 姿态 ← 冲突！
```

### 解决方案：Event-Normalized Query

```python
def query_event_warped_prior(segment, phi, motion_data, segment_bounds):
    """Query desired foot-ball RELATIVE OFFSETS at this event phase.
    
    Returns desired geometric relationships, NOT absolute positions.
    This ensures the prior is about 'where feet should be relative to ball'
    rather than 'where feet should be in world coordinates'.
    """
    i0, i1 = segment_bounds[segment]
    ref_idx = i0 + phi * (i1 - i0)
    ref_state = motion_data.get_state(ref_idx)
    ref_ball = motion_data.get_ball_pos(ref_idx)  # ball pos in original motion
    
    # Desired offsets: foot position RELATIVE TO ball in original motion
    desired_swing_offset = ref_state.swing_foot_pos - ref_ball
    desired_support_offset = ref_state.support_foot_pos - ref_ball
    desired_pelvis_facing = ref_state.pelvis_yaw_relative_to_kick_dir  # relative to kick dir, NOT absolute
    
    return {
        "desired_swing_offset": desired_swing_offset,    # 3D
        "desired_support_offset": desired_support_offset, # 3D
        "desired_pelvis_facing": desired_pelvis_facing,   # 2D
    }

# Reward compares actual vs desired offsets:
# actual_swing_offset = current_swing_foot_pos - current_ball_pos
# actual_support_offset = current_support_foot_pos - current_ball_pos
# error = ||actual_offset - desired_offset||^2
```

> [!IMPORTANT]
> **All weak prior terms must be in ball-relative or kick-direction-relative coordinates, NEVER absolute world coordinates.**
> Weak prior 表达的是 "在这个 event phase，脚应该相对球处在什么几何关系"，
> 而不是 "脚应该去原始 motion 的某个世界位置"。
> pelvis facing 也必须是 relative to desired kick direction，不是 absolute yaw。

**关键原则**：不是 `motion[t]`，而是 `motion[segment, φ_event]`。Retiming 后所有信号语义一致：

```
event says: strike
weak prior points to: original strike posture
contact graph expects: strike
ball relation expects: strike geometry
```

---

## 2. Observation Design (MLP + History)

### Group 1: Current Proprio (~64D)
`base_ang_vel(3) + projected_gravity(3) + joint_pos(29) + joint_vel(29)`

### Group 2: History Buffer (~290D)
```python
history = [
    joint_pos_history,    # 3 frames × 29 = 87D
    joint_vel_history,    # 3 frames × 29 = 87D
    action_history,       # 3 frames × 29 = 87D (includes current last_action)
    last_action,          # 29D (most recent)
]
```
> 50Hz 下 3 frames = 60ms proprio history，足够估计加速度/惯性。

### Group 3: Ball History (~30D)
```python
ball_history = [
    ball_pos_local_history,  # 10 frames × 3D = 30D (pelvis-local XYZ)
]
```
> 10 frames = 0.2s，对 0.03~0.6 m/s 球速足够推断运动趋势。
> 显式 history 代替 LSTM 隐式记忆，更透明可控。

### Group 4: Event Condition (~8D)
```python
event_obs = [
    phase_onehot,           # 4D: approach / prestrike / strike / followthru
    sin(π * φ), cos(π * φ), # 2D: smooth phase progress
    time_to_strike_norm,    # 1D
    time_to_phase_end_norm, # 1D
]
```

### Group 5: Ball-Foot Relation (~24D)
```python
ball_foot_relation = [
    ball_rel_swing_foot,     ball_rel_support_foot,     ball_rel_pelvis,  # 9D
    ball_velocity_local,     desired_kick_dir_local,                      # 5D
    # Contact-ready scalars:
    swing_foot_ball_dist,    swing_ball_longitudinal,   swing_ball_lateral,  # 3D
    support_ball_lateral,    support_ball_longitudinal,                      # 2D
    swing_vel_along_kick,    swing_vel_to_ball_align,                        # 2D
    # + ball_velocity_magnitude for easy thresholding                       # 1D
]  # ~22-24D
```

### Group 6: Event-Warped Weak Prior (~8D)
```python
weak_prior = [
    desired_swing_offset,    # 3D: ref_swing_foot_event - ref_ball_event
    desired_support_offset,  # 3D: ref_support_foot_event - ref_ball_event
    desired_pelvis_facing,   # 2D
]
```

### 总 obs: ~420D

MLP 512→256→128→29 完全能处理这个维度。

---

## 3. Event Phase System

### Motion Segmentation

```
|←── approach ──→|←── prestrike ──→|←── strike ──→|←── followthru ──→|
0              T_a              T_ps            T_s               T_end
```

初始标注（基于已有 kick_frame / kick_end_frame）：
- `T_a = kick_frame - 20` (prestrike = 20 frames before contact)
- `T_ps = kick_frame`
- `T_s = kick_end_frame` (or kick_frame + 8)
- `T_end = motion_length`

> 方案 C: 自动检测 + 手动校正。先用 kick_frame 跑通，同时保存 debug plot，后续用 swing-foot velocity peak / contact force 自动 refine。两条 motion 手动校正成本很低，但 pipeline 要为后续视频数据准备自动分段。

### Event Retiming Augmentation

训练时随机 scale 每个 segment 的持续时间：

```python
@configclass
class EventRetimingCfg:
    enable: bool = False
    approach_scale: tuple = (0.8, 1.3)
    prestrike_scale: tuple = (0.7, 1.5)
    strike_scale: tuple = (0.9, 1.1)      # strike 动作变化小
    followthru_scale: tuple = (0.8, 1.5)
```

Retiming 改的是 event boundary 位置 **AND** weak prior query。motion reference 通过 `motion[segment, φ_event]` 查询，语义始终一致。

Retiming curriculum（从小到大）：

```
v10.1: 不 retime
v10.2a: ±2 frames
v10.2b: ±5 frames
v10.2c: ±10 frames
v10.2d: ±20 frames
```

---

## 4. Reward Restructuring

```python
r = w_prior * r_body_prior          # event-warped 弱姿态先验
  + w_rel * r_foot_ball_relative    # 脚-球空间关系
  + w_cg * r_contact_graph          # contact graph 匹配
  + w_obj * r_ball_outcome          # 球的最终结果
  + w_reg * r_regularization        # 平滑/稳定
  + w_bc * r_bc_to_v3               # v10.1 only: BC distillation
```

### 各阶段权重

| Reward | v10.1 | v10.2 | v10.3 | v10.4 |
|--------|:-----:|:-----:|:-----:|:-----:|
| body_prior | **0.7** | 0.5 | 0.3 | 0.2 |
| foot_ball_rel | 0.2 | 0.4 | **0.7** | 0.7 |
| contact_graph | 0.2 | 0.4 | **0.7** | 0.7 |
| ball_outcome | 50 | 50 | 50 | 50 |
| BC_to_v3 | **strong** | weak→0 | 0 | 0 |
| regularization | 保持 | 保持 | 保持 | 保持 |

> Stage 1 保守：目标是 **preserve v3 能力**，不是泛化。BC loss 确保初始行为接近 v3。

### r_body_prior（event-warped）
- 不再强制 full-body tracking
- 只对 event-warped sparse targets 做 soft matching
- 权重随 stage 递减

### r_foot_ball_relative（核心新 reward）
```
prestrike:
  swing foot 在球后方              → 奖励
  support foot 在球侧后方 (~0.2m)  → 奖励
  pelvis 朝向合理                  → 奖励

strike:
  swing foot 接近球 (dist → 0)     → 高奖励
  foot vel 指向目标 (align > 0.9)  → 高奖励
  contact 在 strike phase 内       → 奖励

followthru:
  follow-through 自然              → 轻奖励
  不摔                             → 奖励
```

### r_contact_graph（event-conditioned）
- Contact graph 作为 **condition input**，不是固定标注
- desired_contact_graph 由当前 event phase 决定
- Reward = actual contact events 与 desired CG 的匹配度

### r_bc_to_v3（Stage 1 only）
```python
# 只在 nominal timing / static ball 上用
L_BC = ||a_new(s_t) - a_v3(s_t_v3_obs)||^2
```
目的：保住 v3 的基础踢球能力，防止新 policy 在冷启动时崩坏。

---

## 5. BC Warm-Start (v3 → v10)

> [!WARNING]
> v3 是 LSTM，v10 是 MLP——架构完全不同，**无法做 weight transfer**。
> 必须通过 BC pretrain 从 v3 teacher rollout 中学习基本踢球行为。

**BC 数据来源**：frozen v3 (LSTM) 在 nominal static ball 环境中的 closed-loop rollout。

```python
# Step 1: Collect teacher rollouts (v3 LSTM runs with proper hidden state)
rollout_data = collect_v3_rollouts(
    env=nominal_static_env, policy=frozen_v3,
    num_episodes=200,
    record_fields=["obs_v10", "teacher_action"]  # obs_v10 includes history buffer
)

# Step 2: Single-step BC pretrain (MLP has no hidden state!)
for obs_v10, a_teacher in DataLoader(rollout_data, shuffle=True, batch=256):
    a_student = pi_v10_mlp(obs_v10)  # MLP, no sequence needed
    loss = ((a_student - a_teacher) ** 2).mean()
    loss.backward(); optimizer.step()
```

> [!IMPORTANT]
> **BC teacher action 必须用 mean action（deterministic），不用 sampled action**。
> `a_teacher = pi_v3.act_inference(obs_v3)` — 避免 student 学到 exploration noise。

> [!WARNING]
> **Covariate shift 问题**：obs_v10 包含 action_history，BC 数据中是 v3 action history，
> 但部署 v10 后 history 变成 v10 action history，造成 distribution shift。
> 解决：(A) v10.1b PPO fine-tune 会自然纠正；(B) 如果 BC loss 低但 rollout 差，做 DAgger：
> 用 v10 rollout 重新收集 states，在这些 states 上 query v3 teacher，加入 dataset 再 BC。

> [!NOTE]
> MLP 的巨大优势：BC 变成标准单步 supervised learning，无需 sequence-level training。
> Teacher (v3 LSTM) 的 hidden state 在数据收集时正常推进，但 student 不需要。
> History buffer 在 env 中维护，作为 obs 的一部分直接输入 MLP。

---

## 6. Training Stages with Pass/Fail Criteria

### v10.1a: Supervised BC Pretrain

```
init:      random MLP weights (no transfer possible from LSTM)
teacher:   frozen v3 LSTM (model_12000)
student:   v10 MLP with new obs (~420D, includes history)
data:      v3 closed-loop rollouts, single-step (obs, action) pairs
loss:      L_BC = ||a_v10_mlp(obs) - a_v3(obs_v3)||²
goal:      student action 近似 teacher action under nominal conditions
duration:  until L_BC converges
```

**Pass criteria:**
- L_BC 收敛（action MSE 足够小）
- 分 phase 的 action MSE：strike-phase MSE 不应显著高于平均
- 部署 student 在 nominal static ball 上 rollout：
  - Kick% ≥ 80%
  - BSpd ≥ 6.5
  - Fall% ≤ 15%
  - Episode length ≥ 80% of v3 average
- 这一步不是最终性能，只确认 student 学到了 teacher 的基本行为

### v10.1b: PPO Fine-Tune (Nominal Recovery)

```
init:      v10.1a BC-pretrained checkpoint
ball:      static, fixed position (same as v3)
retiming:  NONE (nominal event timing)
obs:       new design (~420D, MLP + history)
prior:     event-warped weak prior (timing = original, so = v3 nominal)
reward:    first 20%: body_prior(0.7) + foot_ball_rel(0.05) + contact_graph(0.05)
                      + ball_outcome(50) + BC(strong) + reg
           after stable: foot_ball_rel →0.1→0.2, contact_graph →0.1→0.2
note:      BC-dominated warm PPO 前期，避免 interaction reward 和 BC 打架
goal:      用新 obs/reward 结构完全恢复 v3 静态踢球能力
duration:  ~7000 iterations
```

**Pass criteria:**
- Kick% ≥ 95%
- BSpd ≥ 7.5 m/s
- Fall% ≤ 5%
- CΔ_event ≈ 0 (contact 发生在 strike phase 内)
- Student action 与 v3 action 差异可控

> [!IMPORTANT]
> 如果 v10.1 不通过，后面的 retiming 和 dynamic ball 都不做。这一步只回答一个问题：**新架构能不能不破坏 v3 的基本踢球能力？**

### v10.2: Event Retiming (Static Ball)

```
init:      v10.1b best checkpoint
ball:      static, fixed position
retiming:  curriculum: ±2 → ±5 → ±10 → ±20 frames
prior:     event-warped (now actually retimed!)
reward:    body_prior(0.5) + foot_ball_rel(0.4) + contact_graph(0.4)
           + ball_outcome(50) + reg
goal:      policy 不再绑定固定 frame，能按 retimed event phase 执行
duration:  ~7000 iterations
```

**BC Decay Schedule:**
- BC 只作用在 nominal timing 或 |event_shift| ≤ 2 frames 的 envs
- 不在 ±10/±20 frame retiming 下强迫贴 v3（timing 已不同，强 BC 会冲突）
- 线性衰减：`bc_weight = max(0, bc_init * (1 - progress / decay_steps))`
- 后期完全去掉 BC，只依赖 event/contact/ball rewards

**Pass criteria:**
- event_shift ±5: Kick% ≥ 90%
- event_shift ±10: Kick% ≥ 85%
- BSpd ≥ 7.0 m/s
- **Retiming 验证**（核心指标）：
  - CΔ_event ≈ 0（contact 在 retimed strike phase 内）
  - CΔ_original ≈ event_shift（contact frame 随 retiming 移动）
  - 如果 CΔ_original ≈ 0，说明 policy 还是在原始 kick_frame 踢，没有真正学会 retiming

### v10.3: Interaction-Dominant (Static Perturbation)

```
init:      v10.2 best checkpoint
ball:      static + position perturbation ±15~20cm
retiming:  ±10~20 frames
prior:     event-warped, weight reduced
reward:    body_prior(0.3) + foot_ball_rel(0.7) + contact_graph(0.7)
           + ball_outcome(50) + reg
goal:      interaction relation 主导，support foot geometry 随球位置变化
duration:  ~5000 iterations
```

**Pass criteria:**
- ±15cm offset: Kick% ≥ 90%
- ±20cm offset: Kick% ≥ 85%
- BSpd ≥ 7.0 m/s
- **Support foot geometry 必须随球位变化**（量化指标）：

```
几何相关性检验（在 perturbation grid eval 上计算）：
  d(support_foot_lateral) / d(ball_y_offset)     → 斜率显著 > 0
  d(support_foot_longitudinal) / d(ball_x_offset) → 斜率显著 > 0
  d(swing_foot_contact_pos) / d(ball_pos)         → 斜率显著 > 0
  d(pelvis_yaw) / d(target_dir)                   → 斜率显著 > 0

v3 的这些斜率 ≈ 0（geometry 不随球变）。
v10.3 如果 Kick% 高但斜率仍 ≈ 0，说明还是靠 fixed trajectory tolerance，不是真 adaptive。
```

- **Placement error**（直接对应 weak prior + relation reward）：
  - `||actual_support_offset - desired_support_offset||` 应显著低于 random
  - `||actual_swing_offset - desired_swing_offset||` at contact 应显著低于 random

### v10.4: Dynamic Ball

```
init:      v10.3 best checkpoint
ball:      rolling, sub-staged (very conservative start):
           4a: 0.03~0.10 m/s time-aligned (先验证 event-conditioned timing)
           4b: 0.10~0.30 m/s with arrival jitter
           4c: 0.30~0.60 m/s with position jitter
prior:     minimal
reward:    body_prior(0.2) + foot_ball_rel(0.7) + contact_graph(0.7)
           + ball_outcome(50) + reg
goal:      真正的动态球适应
duration:  ~5000 iterations
```

> [!NOTE]
> v9a 用 0.3~1.0 m/s 造成了 0.4~1.4m 球位变化和严重 timing mismatch。
> 这次从 0.03 m/s 开始，approach phase (~1.4s) 内球仅移动 ~4cm，足以验证
> event-conditioned timing 是否工作，而不至于造成灾难性 mismatch。

**Pass criteria:**
- BSpd recovers from v9a's 4.6 → ≥ 7.0 m/s
- CΔ_event ≈ 0
- Early collision ≤ 3%
- Fall% ≤ 8%
- DirA ≥ 0.9

---

## 7. Implementation Order

```
Phase 1: Infrastructure (1-2 days code)
├── 1.1 Event phase system
│   ├── Motion segmentation (4 phases) in commands
│   ├── Event-normalized phase computation
│   ├── Event-warped motion query function (relative offsets)
│   └── Event retiming augmentation logic
├── 1.2 New observation functions + history buffer
│   ├── History buffer manager (joint_pos/vel ×3f, action ×3f, ball_pos ×10f)
│   ├── event_condition_obs (phase_onehot, sin/cos, t2strike, t2phase_end)
│   ├── ball_foot_relation (raw 3D + contact-ready scalars)
│   └── event_warped_weak_prior (all in relative coordinates)
├── 1.3 New reward functions
│   ├── r_foot_ball_relative (phase-aware)
│   └── r_contact_graph_match
├── 1.4 Teacher rollout + BC utilities
│   ├── collect_v3_teacher_rollouts.py
│   ├── train_v10_bc.py
│   └── eval_v10_bc_rollout.py
└── 1.5 New env config + MLP policy config
    └── G1AnchorEventConditionedKickEnvCfg (v10.1 → v10.4)

Phase 2: BC Pretrain v10.1a + PPO v10.1b (~1-2 days GPU)
Phase 3: Train v10.2 retiming (~7000 iter, 1-2 days GPU)
Phase 4: Train v10.3 perturbation (~5000 iter, 1-2 days GPU)
Phase 5: Train v10.4 dynamic ball (~5000 iter, 1-2 days GPU)
```

---

## 8. v3 → v10 Comparison

| Aspect | v3 (现在) | v10 (目标) |
|--------|----------|-----------|
| 踢球条件 | t = kick_frame | event phase = strike |
| Obs | 160D (motion ref heavy, LSTM) | **~420D** (MLP + history + event + ball relation) |
| Motion reference | 强 tracking (全 body) | event-warped sparse prior |
| Prior query | `motion[t]` | `motion[segment, φ_event]` |
| Contact graph | 固定标注, reward mask | Condition input + retimeable |
| Ball handling | 静态, 固定位置 | 动态, 位置/速度可变 |
| Timing | 固定 (kick_frame) | Event-normalized, retimeable |
| Reward | body_tracking dominant | interaction_relation dominant |
| Support foot | 固定 geometry (不随球变) | **随球位置/timing 主动调整** |
| 泛化 | ±10cm static (tracking tolerance) | Variable timing + position + velocity |
| Latent-ready | No (固定 trajectory) | Yes (event-conditioned primitive family) |
