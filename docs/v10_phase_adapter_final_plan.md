# v10 Phase-Retiming Adapter — Final Implementation Plan

> All design decisions locked. Ready for execution.

---

## Architecture (Final)

```
                    ┌──────────────────────────────────────┐
  real ball obs ──►│  HL Phase Adapter (trainable)         │
  (~30D)           │  MLP(64,32) → Δτ_raw (1D)            │
                   │  EMA + rate limit → Δτ_applied        │
                    └─────────────┬────────────────────────┘
                                  │ Δτ_applied
                                  ▼
                    τ = clamp(τ_prev + 1 + Δspeed, ...)
                                  │
                     ┌────────────┴────────────┐
                     │  motion_lib[τ] (float)   │ ← nlerp for quats
                     └────────────┬────────────┘
                                  │ ref_τ
  nominal ball obs ──►┌───────────┴──────────────────────┐
  (shielded)         │  v3 Frozen Policy (160D → 29D)    │
                     │  LSTM(160, 128, 2 layers)          │
                      └──────────────────────────────────┘
```

### Observation Shielding（关键设计）

| 观察者 | 看到的球 | 原因 |
|--------|---------|------|
| **HL** | 真实 moving ball（pos history, rel feet） | 需要推断球速做 timing |
| **v3** | nominal ball under τ-retimed reference | 防止 LSTM OOD，保持训练分布 |

v3 的 `anchor_ball_polar` 应该用：
```python
# v3 看到的 ball position = motion reference 在 τ 时刻的 nominal contact position
# 不是 sim 中的真实球位
nominal_ball_pos = motion_lib.get_ball_target_pos(tau)
v3_ball_polar = compute_polar(nominal_ball_pos, pelvis_pos, pelvis_yaw)
```

---

## Decision Log

| Question | Decision | Rationale |
|----------|----------|-----------|
| Q1: Training | **Option D**: 1D Δτ action, v3 frozen in env | 最干净，v3 完全不变 |
| Q2: Frequency | **每步输出** + EMA + rate limit | Timing 是精细时序问题，不能 5 步 hold |
| Q3: Interpolation | **Float τ**, linear + nlerp | 必须支持，先 nlerp 后续 slerp |

---

## Δτ Pipeline

```python
# 1. HL raw output
delta_raw = K * torch.tanh(policy_output)  # K=5 frames

# 2. EMA low-pass
alpha = 0.3
delta_smooth = (1 - alpha) * delta_prev + alpha * delta_raw

# 3. Rate limit
max_delta_rate = 1.0  # frame/step
delta_applied = clamp(delta_smooth,
                      delta_prev - max_delta_rate,
                      delta_prev + max_delta_rate)

# 4. Update tau with monotonic clamp
tau_speed = 1.0 + (delta_applied - delta_prev)  # nominal speed=1 + correction
min_phase_speed = 0.2
max_phase_speed = 2.0
tau = tau_prev + clamp(tau_speed, min_phase_speed, max_phase_speed)
tau = clamp(tau, 0, motion_length - 1)
```

### Regularization (normalized)

```python
r_delta = -lambda_delta * (delta_applied / K)**2        # magnitude
r_rate  = -lambda_rate * ((delta_applied - delta_prev) / K)**2  # smoothness
```

建议初始：`lambda_delta = 0.01`, `lambda_rate = 0.005`

---

## tau-space Conversion Checklist

| 系统 | 用 t 还是 τ | 改动 |
|------|:-----------:|------|
| motion command obs (ref state) | **τ** | `get_motion_state(tau)` |
| tracking reward targets | **τ** | `motion_body_pos` etc 用 `motion[tau]` |
| CG reward masks (strike/pre/post) | **τ** | `strike_mask = abs(tau - kick_frame) < window` |
| early_collision判定 | **τ** | `contact && tau < kick_frame - margin` |
| target_point_contact | **τ** | `contact && kick_frame <= tau <= kick_end_frame` |
| post_strike reward | **τ** | `tau > kick_end_frame` |
| visualization/debug | 两者 | `ref marker = motion[tau]`, log both |
| diagnostics | 两者 | `CΔ_wall` + `CΔ_tau` |

### Dual CΔ Logging

```python
CΔ_wall = actual_contact_wall_frame - nominal_kick_frame
CΔ_tau  = tau_at_contact - kick_frame
```

**成功标准**：`CΔ_tau → 0`（即使 `CΔ_wall ≠ 0`）

---

## Float-τ Motion Interpolation

```python
def get_motion_state_float(tau):
    i0 = torch.floor(tau).long()
    i1 = torch.clamp(i0 + 1, max=T - 1)
    alpha = (tau - i0.float()).unsqueeze(-1)

    # Positions, velocities, joint angles: linear
    joint_pos = (1 - alpha) * motion_joint_pos[i0] + alpha * motion_joint_pos[i1]
    body_pos  = (1 - alpha) * motion_body_pos[i0]  + alpha * motion_body_pos[i1]

    # Quaternions: normalized lerp
    q0 = motion_body_quat[i0]
    q1 = motion_body_quat[i1]
    # Ensure shortest path
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    q_interp = (1 - alpha) * q0 + alpha * q1
    q_interp = q_interp / q_interp.norm(dim=-1, keepdim=True)

    return joint_pos, body_pos, q_interp, ...
```

---

## HL Obs Functions (~30D)

| Obs | Dim | Implementation |
|-----|:---:|----------------|
| `ball_pos_local_history` | 15D | Ring buffer of last 5 frames, pelvis frame |
| `ball_rel_swing_foot` | 3D | `ball_pos_w - swing_foot_pos_w` in pelvis frame |
| `ball_rel_support_foot` | 3D | `ball_pos_w - support_foot_pos_w` in pelvis frame |
| `current_phase` | 1D | `tau / motion_length` (τ-based, not t-based) |
| `kick_frame_countdown` | 1D | `(kick_frame - tau) * dt` |
| `prev_delta_tau` | 1D | Previous Δτ applied |
| `base_proj_gravity` | 3D | Standard |
| `base_ang_vel` | 3D | Standard |
| **Total** | **30D** | |

---

## File Changes

### [NEW] `source/.../mdp/phase_adapter.py`
- `PhaseAdapter` class: small MLP (30D→64→32→1D)
- EMA + rate limit + tau update logic
- Per-env `tau` state, `delta_prev` state, `ball_pos_history` ring buffer

### [NEW] `scripts/rsl_rl/train_phase_adapter.py`
- Training script for Option D
- Env wraps v3 frozen policy
- HL action space = 1D (Δτ_raw)
- After HL outputs Δτ → update tau → query motion[τ] → shielded obs → v3 forward → 29D action → step

### [MODIFY] `commands_multi_motion_soccer.py`
- `_compute_soccer_ball_positions()`: time-aligned rolling ball logic
- `_update_command()`: accept external `tau` override (from phase adapter)
- New method: `get_motion_state_float(tau)` for float interpolation
- `desired_contact_frame` property (= nominal_kick_frame for v10a)

### [MODIFY] `observations_anchor.py`
- New: `ball_pos_local_history()`, `ball_rel_swing_foot()`, `ball_rel_support_foot()`
- New: `kick_frame_countdown()`, `current_phase_tau()`
- Modify: `anchor_ball_polar()` → add `use_nominal=True` flag for v3 shielding

### [MODIFY] `soccer_anchor_env_cfg.py`
- New: `G1AnchorPhaseAdapterKickEnvCfg` (v10)
- HL obs group + v3 shielded obs group
- Ball curriculum config (D0→D5)
- Δτ regularization weights

### [MODIFY] `rewards.py` / `kick_detection.py`
- All phase-dependent logic accepts `tau` parameter
- `early_collision_penalty(tau)`, `target_point_contact(tau)`

### [MODIFY] `eval_kick_diagnostic.py`
- Log both `CΔ_wall` and `CΔ_tau`
- Log `delta_tau_mean`, `delta_tau_std` per episode

### [MODIFY] `__init__.py`
- Register `Anchor-PhaseAdapter-Kick-G1-Soccer-RNN-v0`

---

## Training Curriculum

| Phase | Ball | Speed | Arrive Time | Position | K | Iterations |
|-------|------|:-----:|:-----------:|:--------:|:-:|:----------:|
| D0 | static | 0 | exact | exact | 5 | 3k |
| D1 | rolling | 0.03-0.05 m/s | exact | exact | 5 | 5k |
| D2 | rolling | 0.05-0.10 | ±3 frames | exact | 5 | 3k |
| D3 | rolling | 0.05-0.10 | ±5 frames | exact | 10 | 3k |
| D4 | rolling | 0.05-0.15 | ±10 frames | ±5cm | 10 | 5k |
| D5 | rolling | 0.10-0.20 | ±20 frames | ±10cm | 20 | 5k |

### Pass Criteria

| Phase | Kick% | BSpd | CΔ_tau | CΔ_wall | EarlyCol | Fall |
|-------|:-----:|:----:|:------:|:-------:|:--------:|:----:|
| D0 | ≥95% | ≥7.5 | ≤2 | ≤2 | 0% | ≤5% |
| D1 | ≥90% | ≥7.0 | ≤5 | any | ≤2% | ≤8% |
| D2 | ≥85% | ≥6.5 | ≤5 | any | ≤5% | ≤10% |
| D5 | ≥80% | ≥6.0 | ≤5 | any | ≤5% | ≤12% |

---

## Verification: D0 Sanity Check

```
1. Train HL for 3k steps on static ball
2. Eval: Kick% should be ≈ v3 baseline (≥95%)
3. Check: mean |Δτ| should be < 1 frame
4. If Kick% drops significantly → obs shielding bug or tau-space mask error
```
