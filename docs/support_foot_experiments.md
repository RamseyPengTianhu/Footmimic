# 支撑脚落位先验 — 实验总结

## 1. v3_softmask ✅（当前最佳，基线）

**训练**: Stage 1 (model_6999, 纯动作学习) → Stage 2 (训练 5000 步到 model_12000)

**Reward 结构**（Stage 2 在 Stage 1 基础上新增的）:
| Reward | Weight | 职责 |
|--------|--------|------|
| `dynamic_ankle_masking_body_pos` | 1.0 | 动作追踪，CG1 期间释放踢球脚 |
| `time_gated_contact` | **50.0** | CG1 合法触球奖励（核心驱动） |
| `early_collision_penalty` | **-15.0** | CG0 早碰惩罚 |
| `ankle_lock_on_contact` | -0.0 | （已禁用） |
| `interaction_termination` | — | 没踢球则 episode 结束 |
| `curve_offset_range` | — | 球只放在机器人前方 (0.0, 0.4)m |
| 继承: `motion_body_lin_vel/ang_vel` | 0.3 | 降权的速度追踪 |

**结果**:
- **Kick%: 97%**, EarlyCol: 5%, Fall: 15%, BSpd: 4.89 m/s

**v3 的"隐式支撑脚控制"**: v3 没有任何显式的支撑脚 reward，但 97% 的踢球成功率说明 motion tracking (body_pos) 本身就在隐式地引导支撑脚位置和朝向。机器人通过模仿参考动作，自然学会了合理的支撑脚落位。

---

## 2. 后续版本都比 v3 差 — 逐个分析

### v6: 支撑脚位置奖励 ❌
**在 v3 基础上新增**: `support_foot_placement` — dense 正奖励，把支撑脚拉到球旁边特定位置
**结果**: Kick%: 97%, **EarlyCol: 87%**
**失败原因**: 位置奖励在踢球前就激活，等于催促机器人更快靠近球。motion tracking 引导的自然接近节奏被打破，踢球腿在摆腿过程中撞到球。

---

### v71: 稳定性先验 (stable + yaw) ❌
**在 v3 基础上新增**: `support_foot_stability_prior` (weight=0.2)
- dense 正奖励，全程（CG0+CG1）激活
- `r_stable`: 奖励支撑脚低速（vel_std=0.2）
- `r_yaw`: 奖励支撑脚朝向踢球方向（yaw_std=0.6）
- 门控: `pre_ball × no_ball_contact × near_ball × ground_contact`
- `stable_weight=0.3`, `yaw_weight=0.7`

**结果**: **Kick%: 11%**, Miss: 87%, EarlyCol: 0%, BSpd: 7.4, ArgΔ: **+152 帧**
**失败原因**: `r_stable` 奖励支撑脚低速度 → 策略学会了"靠近球后站着不动"。在 CG0 和 CG1 期间都给 dense 正奖励意味着"不踢球也能持续拿 reward"。ArgΔ=+152 表示机器人比参考动作晚 152 帧才接近球，本质上是在"罚站"。当少数情况下踢到球时，球速和方向都非常好（7.4 m/s, DirA=0.98），说明**如果能踢到，支撑脚质量确实提高了**。

---

### v72: 纯 yaw（去除速度惩罚）❌
**修改**: 在 v71 基础上 `stable_weight=0.0`, `yaw_weight=1.0`
**结果**: Kick%: 31%, **EarlyCol: 44%**, Miss: 2%, BSpd: 3.7, Long: -0.75
**失败原因**: 去掉 `r_stable` 后机器人不再罚站（Miss 从 87% 降到 2%），但也失去了接近阶段的速度控制。机器人冲到球前面没刹住车，用身体直接撞飞球。支撑脚位置完全失控（Long=-0.75，离球太远）。

---

### v73: CG0 brake + CG1 release ❌
**修改**: 在 v71 基础上加入 CG phase 门控
- CG0 期间: 激活 brake prior（stable + yaw）
- CG1 期间: 自动关闭（reward=0）
- 恢复 `stable_weight=0.3`, `yaw_weight=0.7`

**结果**: **Kick%: 8%**, Miss: 81%, EarlyCol: 5%, BSpd: **7.8**, Long: -0.21, ArgΔ: **+127 帧**
**失败原因**: CG0/CG1 的结构分离概念正确（EarlyCol 被抑制到 5%），但 **CG0 期间的 dense brake 已经把整个接近策略训成了保守风格**。即使 CG1 释放了约束，策略已经学会了"慢慢走、稳稳停"（Vel=0.07），没有动量完成最后一步踢球。问题不在于 reward 在哪个阶段生效，而在于 dense r_stable 从根本上改变了运动风格。

---

### v74: 纯 sparse bonus（无 brake）❌
**修改**: 
- 关闭 brake prior: `support_foot_placement: weight=0.0`
- 新增: `support_contact_quality_bonus: weight=5.0`
  - 只在 `is_cg1 & ball_contact & correct_foot` 时激活
  - `q_support = ground_contact × r_yaw`（值在 0~1 之间）
  - 最大额外奖励: 5.0（相对于 contact reward 50.0 是 10%）

**结果**: Kick%: 11%, Miss: 2%, **EarlyCol: 66%**, Fall: 25%, BSpd: 6.3, Long: -0.79, ArgΔ: +127
**失败原因**: 与 v3 唯一的差异就是多了 `support_contact_quality_bonus` (weight=5.0)。一个 weight=5 的 sparse bonus 理论上不应该导致 66% 早碰。但从 v3 resume 训练 19000 步后性能全面崩溃。可能的解释：
1. v3 只需要 5000 步就收敛了，额外训练 19000 步本身就是 over-training
2. sparse bonus 虽然小，但改变了 critic 的 value 估计，导致策略在漫长训练中漂移
3. `foot_contact` sensor 的引入可能有未知副作用

---

## 3. 全版本对比总表

| 指标 | v3 基线 | v71 | v72 | v73 | v74 |
|------|:---:|:---:|:---:|:---:|:---:|
| **Kick%** | **97%** | 11% | 31% | 8% | 11% |
| Miss | — | 87% | 2% | 81% | 2% |
| EarlyCol | 5% | 0% | 44% | 5% | **66%** |
| Fall | 15% | 8% | 24% | 6% | 25% |
| BSpd | 4.89 | **7.4** | 3.7 | **7.8** | 6.3 |
| Long | — | -0.22 | -0.75 | -0.21 | -0.79 |
| ArgΔ | — | +152 | +111 | +127 | +127 |

## 4. 核心矛盾

所有实验揭示了一个根本性张力：

- **有 dense brake** (v71, v73): 早碰被抑制（0~5%），但策略变得被动（Miss 81~87%）
- **无 dense brake** (v72, v74): 策略不再被动（Miss 2%），但早碰灾难（44~66%）
- **v3 不需要 brake 就做到了 97% Kick + 5% EarlyCol**：因为 motion tracking 本身提供了隐式的速度和位置控制

## 5. 关键经验

1. **v3 的 motion tracking 是隐式的支撑脚控制器** — 不需要显式 reward
2. **任何 dense 支撑脚 reward 都会破坏 v3 的接近节奏** — 不论是位置(v6)还是速度(v71/v73)
3. **即使极小的 sparse bonus (v74, weight=5) 也可能导致策略漂移** — 可能与训练步数有关
4. **踢球质量(BSpd)和踢球成功率(Kick%)目前不可兼得** — v71/v73 踢到时球速 7.4~7.8，但踢中率只有 8~11%
5. **20N 接地力阈值是正确的** — 力传感器分布验证为双峰（0 或 150+N）
6. **early_collision_penalty (weight=-15) 不足以独立防止早碰** — 在没有 brake prior 的情况下（v72/v74），早碰率仍然 44~66%