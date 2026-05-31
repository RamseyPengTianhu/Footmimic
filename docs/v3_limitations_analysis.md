# v3 Kick Policy 现状分析与局限性

> **Baseline**: v3 (cg_v3_softmask, model_12000) — Kick%=97%, BSpd=8.0, DirA=0.97, Fall%=3%

---

## 1. 数据与标注瓶颈

### 1.1 Contact Phase 手动标注不精确

Contact phase（kick_frame / kick_end_frame）由人工标注，存在系统性误差：
- 标注基于视觉判断，无法精确对齐物理接触帧
- 不同 motion 的标注标准不一致
- CG（Contact Graph）奖励的触发时机直接依赖这些标注——标注偏早会导致 policy 在助跑阶段就尝试发力，标注偏晚则错过最佳击球窗口
- 下游影响：`early_collision_penalty` 和 `target_point_contact` 的边界判断全部受此影响

### 1.2 Motion 数据量极少

仅有 **2 条** kick motion（hmr4d_1, hmr4d_4），均为右脚踢球：
- 极易 overfitting 到特定轨迹——policy 本质上是在"记忆"两条固定的运动序列
- 无左脚踢球数据，Foot%=100% 看似完美，实则是数据偏差
- 两条 motion 的 approach 路径、timing、击球角度高度相似，泛化空间极窄

---

## 2. 空间泛化能力评估（Perturbation Grid）

为量化 v3 的泛化边界，在 eval 时对球位置施加确定性 XY 偏移（球仍静止），5×3 grid，10 episodes/条件。

### 2.1 Kick% Heatmap

```
           y=-0.10    y=0.00    y=+0.10
x=-0.10      95%       100%       95%
x=-0.05      95%       100%       95%
x= 0.00      95%        90%      100%
x=+0.05      95%       100%      100%
x=+0.10      90%       100%      100%
```

### 2.2 Success% (outcome) Heatmap

```
           y=-0.10    y=0.00    y=+0.10
x=-0.10      95%        95%       95%
x=-0.05      95%        95%       95%
x= 0.00      95%        90%      100%
x=+0.05      95%       100%      100%
x=+0.10      90%       100%      100%
```

### 2.3 Fall% (outcome) Heatmap

```
           y=-0.10    y=0.00    y=+0.10
x=-0.10       5%         0%        5%
x=-0.05       5%         0%        5%
x= 0.00       5%        10%        0%
x=+0.05       5%         0%        0%
x=+0.10      10%         0%        0%
```

### 2.4 BSpd (peak ball speed, m/s) Heatmap

```
           y=-0.10    y=0.00    y=+0.10
x=-0.10      7.7        7.9       7.5
x=-0.05      7.7        7.7       7.3
x= 0.00      7.8        7.2       8.3
x=+0.05      7.3        8.0       8.0
x=+0.10      7.5        8.2       7.8
```

### 2.5 DirA (ball direction alignment) Heatmap

```
           y=-0.10    y=0.00    y=+0.10
x=-0.10     0.95       1.00      0.95
x=-0.05     0.95       1.00      0.95
x= 0.00     0.95       0.90      1.00
x=+0.05     0.95       1.00      1.00
x=+0.10     0.90       1.00      1.00
```

### 2.6 Early Collision & Miss

所有 15 组条件：**Early Collision = 0%，Miss = 0%**。唯一失败模式为 Fall。

### 2.7 Support Foot Geometry（at contact）

| Condition | Lat | Long | YawE | Gnd% |
|-----------|:---:|:----:|:----:|:----:|
| (-0.10,-0.10) | 0.23 | -0.17 | 0.33 | 93% |
| (-0.10, 0.00) | 0.25 | -0.23 | 0.44 | 91% |
| (-0.10,+0.10) | 0.23 | -0.20 | 0.36 | 94% |
| (-0.05,-0.10) | 0.21 | -0.23 | 0.37 | 94% |
| (-0.05, 0.00) | 0.27 | -0.21 | 0.38 | 94% |
| (-0.05,+0.10) | 0.22 | -0.16 | 0.33 | 95% |
| ( 0.00,-0.10) | 0.22 | -0.21 | 0.38 | 95% |
| ( 0.00, 0.00) | 0.23 | -0.17 | 0.30 | 88% |
| ( 0.00,+0.10) | 0.23 | -0.22 | 0.39 | 98% |
| (+0.05,-0.10) | 0.23 | -0.19 | 0.31 | 93% |
| (+0.05, 0.00) | 0.23 | -0.19 | 0.34 | 99% |
| (+0.05,+0.10) | 0.24 | -0.22 | 0.37 | 97% |
| (+0.10,-0.10) | 0.20 | -0.21 | 0.37 | 88% |
| (+0.10, 0.00) | 0.24 | -0.20 | 0.37 | 94% |
| (+0.10,+0.10) | 0.24 | -0.22 | 0.40 | 98% |

### 2.8 Grid 实验结论

1. **v3 在 ±10cm 内 Kick% 稳定在 90~100%**，无 early collision 或 miss
2. **Support foot geometry 在所有条件下几乎不变**（Lat=0.20~0.27m, Long=-0.16~-0.23m）——policy 走的是同一条轨迹，泛化来自 motion tracking 的固有容忍度，而非 policy 主动适应
3. **BSpd 稳定在 7.2~8.3 m/s**，无系统性退化
4. Baseline (0,0) 的 Kick%=90% 反而低于多数偏移条件，可能为样本量波动

---

## 3. Task 能力局限

### 3.1 仅支持定点踢球（Static Kick）

当前 task 设定是：球静止 → 助跑 → 一次性踢出。这对应真实足球中的**定位球（set piece）**场景。

缺失的核心能力：
- **盘带（Dribbling）**：连续小幅触球控制球的行进方向，需要多次精确触球
- **接球后踢（First Touch）**：球滚过来时迎球踢出，需要 timing 适应
- **带球过人**：边跑边控球，需要步态和触球的协调

### 3.2 没有连续交互能力

当前 episode 结构是"一锤子买卖"：
- `interaction_fail` termination 在 motion 结束时检查——球没动就判定失败
- 没有多次触球的奖励机制
- 没有"控球→调整→再踢"的 episode 设计

---

## 4. 动态球处理能力缺失

### 4.1 静态球假设

训练时球始终静止在参考位置，eval 时球也是静止的。Policy 从未见过：
- 球在助跑过程中位置发生变化
- 球有初始速度（滚动/弹跳）
- 球被碰到后的位置变化

### 4.2 滚动球实验（v9a）失败分析

v9a 加入 0.3~1.0 m/s 朝机器人方向滚动的球，结果：

| 指标 | v3 (静态) | v9a (滚动) | 变化 |
|------|:---------:|:----------:|:----:|
| Kick% | 97% | 84% | ↓13% |
| BSpd | 8.0 | 4.6 | ↓43% |
| KickV | 6.7 | 4.7 | ↓30% |
| Early Col | 0% | 5% | ↑ |
| Fall% | 3% | 11% | ↑8% |
| DirA | 0.97 | 0.90 | ↓ |
| CΔ (contact timing) | -1 | +62 | timing 错位 |

**失败原因**：
- 球以 0.3~1.0 m/s 滚向机器人，approach phase（~1.4s）内球移动 0.4~1.4m
- 球直接滚进助跑路径 → **误触**（early collision）
- Motion tracking reward 强制 policy 走固定轨迹，无法偏离避球
- 没有 ball velocity observation → 无法预判拦截点
- CΔ=+62 帧：接触发生在预期 kick_frame 之后 62 帧，timing 完全错位
- BSpd 下降 43%：球朝机器人运动，相对冲击速度被部分抵消

### 4.3 根本原因：Motion Reference 与动态环境的冲突

当前架构的核心假设是"环境配合 motion reference"——球放在轨迹终点，robot 走到就踢。当环境变得动态（球会移动），这个假设被打破：
- Policy 被 tracking reward 锁定在固定轨迹上
- 即使 policy 通过 `anchor_ball_polar` 看到球位置变了，也无法偏离轨迹去适应
- 这不只是 observation 的问题，而是 **observation insufficiency + reference-locked reward architecture 的共同问题**。当前 160D obs 仅含 `anchor_ball_polar`（3D），缺少 ball velocity、ball-relative foot position 和 kick context；但即使补全 observation，如果 tracking reward 仍然强锁 reference trajectory，policy 也未必会利用这些信息去主动改变轨迹

---

## 5. Reward Engineering 冲突

### 5.1 已实验的辅助 reward 及结果

| 实验 | Reward | 结果 | 失败原因 |
|------|--------|------|----------|
| v71~v74 | 立足脚平行站位 | Kick% ↓ | 与 motion tracking 冲突，policy 为摆站位牺牲 approach 质量 |
| v71~v74 | Orientation 朝向目标 | Kick% ↓ | 与 `pelvis_orientation` reward 冲突 |
| v75 | 踢球脚踝锁定 | 无改善 | 踝关节力矩小，锁不锁对 BSpd 影响有限 |
| v8a | 踢后站稳 (post_strike) | Kick%↓, Fall%↑ | post-strike metrics improved, but Kick% decreased and Fall% increased due to PPO credit leakage |

### 5.2 结构性原因

所有辅助 reward 失败的共同模式：
1. **与 motion tracking 的 reward 梯度方向冲突** — body_pos/ori/foot_pos 权重大（1.0），新 reward 权重小（0.05~0.3），但 PPO 的优势函数会放大小 reward 的影响
2. **作用时序重叠** — 辅助 reward 在 contact 前生效时，会干扰 approach 质量；在 contact 后生效时（如 post_strike），通过 value function 反传仍然影响 contact 前的策略
3. **Motion reference 已经隐含了最优策略** — 如果 reference 本身就有合理的站位和朝向，tracking reward 已经在引导 policy 做对的事。额外的 dense reward 是冗余甚至矛盾的信号

---

## 6. Observation 与架构缺陷

### 6.1 缺少关键 observation

当前 policy obs（160D）包含 `anchor_ball_polar`（距离+方位角，3D），但缺少：
- **Ball velocity**（球速，3D）：无法预判球的运动轨迹
- **Ball-relative foot position**（球相对于每只脚的位置，6D）：无法精确调整步伐
- **Kick context**（踢球侧+phase，2D）：phase progress 需要 policy 自己从 time 推断

### 6.2 加新 obs 的代价

加入新 observation 会改变 policy input 维度（160D → 170D+），导致：
- Stage 1 checkpoint 无法直接加载 → **必须重训 Stage 1**
- Stage 1 训练本身需要大量计算资源和时间
- 风险：新 obs 可能引入噪声，如果 reward 没有正确引导 policy 使用新信息，等于白加

### 6.3 RNN 的理论能力与实际局限

LSTM（2层，hidden=128）理论上可以从 `anchor_ball_polar` 的时序变化推断球速。但：
- 需要足够长的训练（v9a 12k steps 不够）
- 推断精度不如直接给 velocity obs
- RNN 容易遗忘——长 approach phase（70+ 帧）后的信息可能丢失

---

## 总结：当前瓶颈优先级

| 优先级 | 瓶颈 | 影响 | 解决路径 |
|:------:|------|------|----------|
| P0 | Motion data 太少（2条） | 泛化上限低 | 采集/生成更多 kick motions |
| P0 | 静态球假设 | 无法处理真实场景 | 渐进式 ball randomization |
| P1 | Reward 架构冲突 | 辅助 reward 无法生效 | 重新设计 reward hierarchy 或 curriculum |
| P1 | 缺少 ball velocity obs | 无法适应动态球 | 加 obs + 重训 Stage 1 |
| P2 | Contact phase 标注不精确 | CG reward 触发偏差 | 自动化标注 / learned contact detection |
---

## 7. 架构演进：从 Hard Tracking 到 Semantic Motion Prior

### 7.1 核心冲突：Tracking 权重过高与环境变化的矛盾

当前 v3 架构最根本的局限在于 **motion reference 被当成了答案，而不是先验**。

原始 tracking 的含义是：

```text
每一帧身体都要像 reference[t]
kick_frame 附近才是合法 contact
```

但真实想要的是：

```text
动作风格、身体协调、接触机制要像 reference
具体 timing、脚落点、接触位置要根据球的位置和状态调整
```

因此 v3 在球位偏移、步态误差、滚动球场景下容易出现 reward 冲突：
1. Policy 为了 tracking 贴住固定轨迹，不能主动调整击球几何。
2. CG 只描述 reference 的时间窗口，不描述当前状态是否真的适合踢。
3. 如果球的真实可踢时机偏离 `kick_frame`，policy 要么错过球，要么牺牲 tracking。

### 7.2 目标范式：Reference as Prior

后续版本的目标不是删除 motion reference，而是把它降级为 prior：

| 类型 | 含义 | 风险/收益 |
|------|------|-----------|
| Hard tracking | 必须复现 reference 每一帧 | 稳定，但 fixed timing / fixed geometry |
| Weak motion prior | 当前状态落在合理运动流形 | 允许根据球调整 |
| Skill prior | 当前动作像 running / plant / strike / follow-through | 能区分跑步和踢球 |
| Interaction prior | 当前触球方式像有效射门 | 防止跑步撞球或乱碰球 |

也就是说，reference 不应该告诉 policy “答案是哪一帧”，而应该告诉 policy “什么样的运动是合理的”。

### 7.3 v36a: Semantic Prior + Phase-Free Contact Quality

v36a 的 task 是：

```text
Anchor-V36-Kick-G1-Soccer-RNN-v0
```

训练起点与 v3 相同：

```text
load_run = 2026-04-10_02-11-14
checkpoint = model_6999.pt
run_name = v36a_semantic_prior_from_v3_start
```

v36a 保持 observation / policy shape 不变，因此可以直接从 v3 checkpoint resume。主要改动是：

1. **Phase-modulated tracking**
   - approach 保持较强 tracking，保证跑步稳定性。
   - prestrike / strike 降低 full-body tracking。
   - `motion_foot_pos` 置 0，避免 exact foot position 与球位置调整冲突。
   - anchor orientation / body velocity tracking 降权。

2. **Phase-free contact quality**
   - 取消原始 `target_point_contact` 和 `early_collision_penalty`。
   - 引入 `D_strike` gating：触球是否像 strike，由 discriminator 判断。
   - ball speed / direction reward 乘上 `D_strike`，希望奖励“像踢球的有效触球”。

3. **Phase-free interaction termination**
   - 不再要求必须在 `kick_end_frame` 附近触球。
   - episode 末尾只检查球是否被有效踢动。

### 7.4 v36a 结果与暴露的问题

从 TensorBoard scalar 看，v36a 并不是训练崩溃。到 `model_11000.pt`：

| 指标 | 数值 |
|------|:----:|
| Mean reward | 25.80 |
| Kick success rate | 0.979 |
| Expected kick success rate | 0.979 |
| Ball direction angle | 0.172 rad |
| `v36_gated_direction` | 0.976 |
| `v36_strike_contact` | 0.350 |
| `v36_gated_ball_speed` | 0.338 |

但是 `eval_kick_diagnostic.py` 暴露出更关键的定性问题：

| 指标 | Aggregate |
|------|:---------:|
| Kick% | 92% |
| Fall% | 12% |
| EarlyCt | 5.6 frames |
| BSpd | 6.7 m/s |
| DirA | 0.95 |
| Foot% | 100% |
| Contact D-score mean | 0.933 |
| `contact_frame - kick_frame` mean | +103.6 frames |
| `argmin_frame - kick_frame` mean | +152 frames |

结论：

```text
v36a 能把球踢出去，但没有真正解决 strike timing 与 ball geometry 的绑定问题。
```

具体失败模式：

1. **Strike 动作和球的位置脱钩**
   - policy 可能在 reference strike 时间附近提前挥腿。
   - 如果球不在可踢区域，就会空挥。
   - 后面再通过晚接触把球碰出去，仍然能拿到 outcome reward。

2. **Contact window 过宽**
   - v36a 不再限制 contact timing，导致 `t >= kick_frame` 后很晚的接触也能拿到较高奖励。
   - 这解决了 hard CG 的僵硬问题，但引入了 “late fallback contact” 漏洞。

3. **D-strike 只判断接触瞬间，不判断准备过程**
   - `d_at_contact = 0.933` 说明接触瞬间看起来像 strike。
   - 但它不能保证 contact 发生在合理的 ball-ready 状态，也不能惩罚之前的空挥。

4. **Strike morphology 约束过松**
   - swing foot tracking 在 strike 近似为 0，释放了调整空间。
   - 但也削弱了“完整踢球动作”的约束，policy 容易学成结果主义的撞击策略。

### 7.5 v36b: Ball-Ready Gated Semantic Prior

v36b 的 task 是：

```text
Anchor-V36B-Kick-G1-Soccer-RNN-v0
```

v36b 不是简单堆 reward，而是在修 v36a 的 credit assignment：

```text
v36a:
  只要接触瞬间像 strike，并且球结果好，就给高 reward

v36b:
  必须 ball-ready、support-foot planted、timing 合理、D_strike 高，才给高 contact reward
```

核心 contact quality：

```text
quality = D_strike^gamma
        * ready_score^ready_gamma
        * timing_gate^timing_gamma
        * correct_foot
```

其中 `ready_score` 包含：
- kick foot 与 ball 的 XY 距离
- kick foot 高度是否在可踢区间
- support foot 相对 ball 的 lateral / longitudinal 位置
- support foot 速度是否足够小
- support foot 高度是否接近 planted
- support foot yaw 是否朝向 kick direction

`timing_gate` 的作用不是回到 hard CG，而是允许 retiming 但抑制极端早/晚接触：

```text
kick_frame - 20 到 kick_frame + 55 附近保留较高信用
更早或更晚的接触 reward 逐渐衰减
```

新增关键 reward：

| Reward | 作用 |
|--------|------|
| `v36b_strike_ready_prior` | 接触前鼓励进入 ball-ready + plant 状态 |
| `v36b_empty_swing_penalty` | 惩罚脚速很高但脚离球仍远的空挥 |
| `v36b_strike_contact` | 只奖励 D-score、ready、timing、correct foot 都合理的触球 |
| `v36b_invalid_contact_penalty` | 对低质量触球给负信用 |
| `v36b_gated_ball_speed` | 只有高质量接触后才奖励出球速度 |
| `v36b_gated_direction` | 只有高质量接触后才奖励出球方向 |

同时，v36b 对 tracking schedule 做了小幅回收：
- strike 阶段 swing foot multiplier 从 0 提到 0.15，避免完全失去踢球腿 morphology。
- support foot multiplier 提到 0.85，强调 plant 质量。
- direction reward 权重从 v36a 的 30 降到 15，避免方向 reward 过早饱和并主导学习。

v36b 想验证的不是“手写 reward 是否最终最优”，而是：

```text
event-gated semantic prior 是否能同时减少提前空挥、晚接触和 early contact，
并保持 ball outcome。
```

如果 v36b 成功，预期 diagnostic 改善应体现在：
- `contact_frame - kick_frame` 从 +100 帧明显下降。
- `argmin_frame - kick_frame` 下降。
- `EarlyCt` 下降。
- `empty_swing_penalty` 逐步降低。
- `v36b_ready_score` 在触球前升高。
- Kick% / BSpd / DirA 不明显崩掉。

### 7.6 v36b2 之后暴露的新问题：First Attempt 不受约束

v36b2 的 phase-based eval 看起来已经明显改善：

```text
Kick% = 92%
success = 90%
BSpd = 6.0 m/s
DirA = 0.90
CΔ ≈ +13 frames
```

但视频显示动作仍然怪：policy 经常先做一次空挥，后面才把球碰出去。

因此新增了 `eval_kick_attempt_diagnostic.py`，把一次 kick 拆成：

```text
first strike attempt
hit within attempt window
late fallback contact
```

v36b2 的 attempt-based 结果：

```text
Clean% = 57%
Late%  = 40%
Empty% = 0%
```

这说明旧 eval 和旧 reward 都仍然偏结果主义：

```text
只要最后球速和方向好，就容易被算作成功；
但第一次真正的 strike attempt 不一定命中球。
```

因此 v36b4 改成 **attempt-window credit assignment**：

```text
第一次高速、朝球方向的 swing 启动 attempt；
contact 必须在 attempt_window 内发生才拿 strike/outcome reward；
如果 attempt window 过了还没接触，标记 missed attempt；
missed attempt 后的 late fallback contact 不再拿球速/方向奖励，并受到 invalid contact 负信用。
```

对应 task：

```text
Anchor-V36B4-Kick-G1-Soccer-RNN-v0
```

v36b4 不是为了追求更高旧 success，而是为了让 `Clean%` 替代旧 `success` 成为主指标。

### 7.7 与 VAE / AMP-style Prior 的关系

v36b 不是 VAE 的替代品。它更像是一个 **event/contact scaffold**，用来修正当前 reward 逻辑：

```text
什么时候该踢？
球是否在可踢区域？
支撑脚是否站稳？
这次 contact 是否有效？
```

而 VAE / AMP-style prior 更适合学习：

```text
什么叫 approach motion
什么叫 plant motion
什么叫 strike motion
什么叫 follow-through motion
```

因此两者关系应该是：

```text
现在:
  手写 ready_score / plant_score / strike-style gate

后续:
  用 phase-conditioned VAE 或 AMP discriminator 替换部分手写 motion prior
  但保留 ball-ready gate、contact outcome、empty-swing/invalid-contact 逻辑
```

换句话说，VAE 可以让 motion prior 更宽、更数据驱动；但 VAE 本身并不自动解决 contact timing 和 ball geometry。即使后续上 VAE，仍然需要类似 v36b 的 event-conditioned contact credit。

### 7.8 当前路线判断

v36a 的价值在于证明：

```text
降低 hard tracking 后，policy 仍然能恢复踢球成功率。
```

v36a 的失败在于证明：

```text
如果没有 ball-ready gate，policy 会学到结果主义触球，而不是可靠的 strike skill。
```

v36b 的目标是验证：

```text
把 strike prior 与 ball-ready geometry 绑定后，
是否能让 policy 学会“等球、站稳、再踢”，
而不是“到了 reference strike 时间就挥腿”。
```

---

## 7. V3 Video Retrain 实验——数据敏感性的最终证明

> **实验**: v3_video_retrain (model_18000)
> **数据**: 8 条 motion（6 条 soccerkicks2 视频重建 + 2 条原始 hmr4d MoCap）
> **评估**: `eval_kick_attempt_diagnostic`（严格 attempt-based）

### 7.1 实验结果

```
Motion                               Att% Clean% Empty% Late% NoAtt% Fall% |  BSpd  DirA
------------------------------------------------------------------------------------------
11_freekick1_unitree_g1_right         75%    25%     5%   45%    25%   40% |   6.1  0.69
16_penalty1_unitree_g1_right         100%    25%     0%   70%     0%    0% |   9.1  1.00
18_freekick_unitree_g1_right          95%     5%    15%   75%     5%   30% |   7.3  0.80
1_freekick_unitree_g1_right           80%    15%     0%   55%    20%   30% |   6.6  0.77
4_penalty5_unitree_g1_right           90%    40%     0%   30%    10%   15% |   7.6  0.85
5_penalty5_unitree_g1_right           75%    30%     0%   45%    25%   25% |   6.9  0.75
hmr4d_1_unitree_g1_compatible_righ    90%    10%    10%   65%    10%   20% |   8.4  0.93
hmr4d_4_unitree_g1_compatible_righ    65%    20%    10%   35%    35%   10% |   8.2  0.92
AGGREGATE                             84%    21%     5%   52%    16%   21% |   7.5  0.84
```

与 V3 baseline（2 条 hmr4d，旧 phase-based 指标 Kick%≈97%）相比，严格 attempt-based eval 下新模型只有 **Clean%=21%**，Fall%=21%。注意这里不是完全同口径比较：旧 baseline 也应该用 `eval_kick_attempt_diagnostic.py` 重新测一次。但即使只看新模型内部，52% Late Fallback 已经说明学习目标明显偏掉。

### 7.2 失败模式分析

**52% Late Fallback 是主要失败模式**——机器人第一次挥腿踢空，靠后续偶然碰撞把球蹭走。

ATTEMPT / CONTACT GEOMETRY 对比：

```
Outcome            N |  ADst  ACls   AKV |  sLat  sLong   sYaw    sV |  CDly  BSpd
----------------------------------------------------------------------------------
clean_success     34 |  0.81  3.69  4.25 |  0.39  -0.11   0.65  1.58 |    8   9.03
late_fallback     84 |  0.83  3.84  4.26 |  0.54  -0.50   0.80  0.22 |   38   9.03
```

关键差异：
- **CDly**: 8 vs **38 frames** — Late Fallback 的第一次 attempt 没在窗口内命中，真正触球晚了很多
- **sLong**（支撑脚相对球纵向位置）: -0.11 vs **-0.50** — Late Fallback 时支撑脚/身体站位更靠后，strike 几何还没进入可击球区域
- **sV**（support foot speed）: 1.58 vs **0.22** — Late Fallback 不是“脚没有挥”，而是支撑脚已经更稳定但第一次 swing 没打到球；后续接触才把球踢走
- **ADst / ACls / AKV 几乎相同** — clean 与 late 都检测到了高速朝球的第一次 swing，因此问题不是没有 strike attempt，而是 **attempt 与 ball-contact geometry 没有对齐**

### 7.3 逐 Motion 原因诊断

| Motion | 核心问题 | 原因 |
|---|---|---|
| `18_freekick` (Clean 5%) | AΔ=-6，approach 阶段就误触发 attempt | 原始视频是斜向助跑，yaw 修正后走路弧线仍然导致脚在错误位置摆动 |
| `11_freekick1` (Fall 40%) | 重心轨迹不稳定 | HMR 重建的骨盆位置噪声大，tracking reward 要求复现不稳定的重心轨迹 |
| `16_penalty1` (Late 70%) | 脚踝末端位置偏移大 | HMR 对脚踝末端的重建精度低（~10-15cm误差 vs MoCap ~2-3cm） |
| `hmr4d_1/4` (退步到 10-20%) | 新数据互相干扰 | Policy 在 8 条 motion 间泛化时学了"平均化"摆腿，对每条单独的精度都下降 |

### 7.4 根本局限性总结

本次实验从数据侧暴露了 V3 框架（rigid trajectory tracking + CG）的一个关键特性：

```text
V3 不是完全不能用视频重建数据；
原始 hmr4d_1 / hmr4d_4 就说明，只要数据处理、球位推断和 kick_frame 标注足够自洽，
hard tracking 仍然可以踢得很好。

但 V3 对这种“reference-contact-environment 几何一致性”的要求非常高。
新增数据表现差，更准确地说是：当前新增数据的处理质量/标注/球位推断还没有达到 V3 所需的精度。
```

#### 1. 对 Motion 数据的精度要求极高

V3 的 tracking reward 要求机器人精确复现参考轨迹的每一帧关节位置/朝向，并且要求 reference 的踢球脚轨迹、球位置、`kick_frame` 在几何上对齐。这意味着：
- 不一定只能用 MoCap；**经过良好处理的 HMR/video reconstruction 也可以用**，原始两条 hmr4d motion 就是例子
- 但可用数据必须满足较高的几何一致性：swing foot arc 要经过球，`kick_frame` 要接近 reference 中的真实触球时刻，支撑脚/身体姿态不能明显漂移
- 如果脚踝末端误差、球位推断误差或 contact 标注误差达到 10cm 量级，就足以让 hard tracking 与 contact reward 发生冲突
- 即使脚踝位置误差只有 5cm，乘以 tracking weight 后也会在 reward landscape 中产生不可忽略的梯度信号，把 policy 推向错误方向

#### 2. 对 Approach Geometry 的假设太强

V3 隐式假设所有 motion 的助跑路径是**直线、正面、朝向 -Y**：
- 球放置位置基于 `first_anchor → last_anchor` 向量计算（`_compute_soccer_ball_positions`）
- Destination 硬编码在 `(0, -5, 0)` 即正前方
- 任何斜向助跑、弧线跑动的 motion 都会导致球位与脚摆弧线不匹配
- 强制 normalize yaw 只能对齐全局朝向，无法修复运动内部的弧线轨迹几何

#### 3. 数据量增加反而导致性能下降

**这违反了 "More Data = Better" 的一般直觉**：
- 2 条处理较好、几何自洽的 hmr4d motion → Kick% 97%
- 8 条混合质量 motion → **Clean% 21%**
- 原有的 hmr4d 数据也从 ~97% 退步到 10-20%
- 根因不是“数据多本身有害”，而是新增 motion 的处理质量、球位推断、approach geometry、`kick_frame` 标注不够一致，导致同一个 rigid tracking objective 下出现冲突信号
- 在 V3 范式下，**未经筛选/校准的 motion 多样性** 会变成冲突信号；高质量且几何一致的数据仍然可能提升性能

#### 4. 框架可扩展性受数据处理质量强约束

| 维度 | V3 的要求 | 现实 |
|---|---|---|
| 数据精度 | 脚端轨迹、base 轨迹要和仿真球位自洽 | 部分视频重建脚端误差可能达到 10cm 量级 |
| 助跑路径 | 球位生成要匹配该 motion 的 approach geometry | 真实踢球有斜跑、弧线跑，不能只靠 first/last anchor 推球 |
| 标注依赖 | `kick_frame` 要接近 reference 真实触球时刻 | 视频数据标注误差会直接改变 CG reward |
| Motion 数量 | 多条 motion 需要质量一致、几何一致 | 未筛选数据会互相干扰 |
| 左/右脚 | 需要明确标注并在 reward 中动态选择 kick/support foot | 写死右脚会污染 left-kick 数据 |

### 7.5 结论

> V3 的 97% Kick Rate 说明：在少量、处理较好、几何自洽的 reference 上，hard tracking + CG 可以得到很强的定点踢球策略。
> 但这也说明 V3 对数据处理质量非常敏感：当新增 motion 的脚端轨迹、球位推断、approach geometry 或 `kick_frame` 标注不够一致时，性能会快速退化。
>
> 因此，新增视频数据不能直接无筛选地当作 hard tracking answer。
> 更合理的路线是：
> 1. 先做 offline motion diagnostic，筛出 reference swing arc、球位、`kick_frame` 自洽的数据；
> 2. 高置信数据可以继续用于 V3/v3-style baseline；
> 3. 低置信但动作风格有价值的数据更适合进入 CVAE / AMP-style motion prior，
>    学习“怎么踢球”的动作分布，而不是逐帧复现。 

---


## 8. LATENT-Style Distillation Pipeline

> **目标**：解决 V3 对 motion reference 数据精度和几何一致性的强依赖（Section 7.4），
> 将 v3 teacher 的能力压缩到 latent action space，最终实现不依赖精确 motion reference 的踢球策略。

### 8.1 动机

V3 的核心局限（Section 7.4）在于对 reference-contact-environment 几何一致性的极高要求。
[LATENT 论文](https://arxiv.org/abs/2407.18106) 提供了一条路线：
1. 先用 teacher 的 rollout 数据学一个 **latent action space**（CVAE）
2. 通过 **online distillation (DAgger)** 保证 student 在自己的分布上有效
3. 在 latent space 上训练 **high-level PPO**，用 LAB（Latent Action Bounds）约束探索范围

### 8.2 Pipeline 结构

```
Stage 1: v3 softmask teacher          obs_v3 (160D) → LSTM → action (29D), Kick%=97%
Stage 2B: Online Distillation (DAgger) Encoder + Decoder + Prior (CVAE, z_dim=16)
Stage 3: PPO + LAB                     PPO(obs) → u(16D) → LAB → z → D(obs,z) → action
```

### 8.3 DAgger obs_mode 实验总览

| 模式 | Decoder/Prior 输入 | 维度 | Prior Kick% | Prior Fall% | 说明 |
|---|---|---|---|---|---|
| `full` | obs_v3 (含 58D motion ref) | 160D | **96%** | **18%** | 天花板，需要 reference |
| `task` | proprio only | 99D | 44% | 78% | 太少信息 |
| `task_features` | proprio + 22D ball-foot | 121D | 66% (best) | 50% (best) | 有提升但不稳定 |

### 8.4 Stage 4: Task-Only Obs PPO

> PPO 不看 motion reference，只靠 body state + ball info 做决策，decoder/prior 保持用完整 obs_v3。

| 指标 | v3 Teacher | Stage 3 Full Obs | **Stage 4 Task Obs** |
|---|---|---|---|
| **Clean%** | 21% | 68% | **82.5%** |
| Kick% | 97% | ~98% | **97.9%** |
| Fall% | 3% | 35% | **1.0%** |
| BSpd (clean) | 9.03 | 7.53 | **8.30** |

**核心结论**：PPO 不需要 motion reference 即可做出高质量踢球决策。

### 8.5 task_features 结构性分析

| Stage | Decoder 输入 | Kick% | Fall% | BSpd |
|---|---|---|---|---|
| Stage4 ref-cond | proprio + **58D motion ref** | 97.9% | 1% | 8.30 |
| Stage5 ref-free | **proprio only** | 44% | 78% | — |
| Stage6 task_feat | proprio + **22D ball-foot** | 61% | 43% | 6.77 |

四个结构性原因：
1. **22D 不含时序/phase**：同一个 ball-foot 几何可能对应 approach/plant/swing/follow-through
2. **Reward mismatch**：tracking reward 要求跟踪 reference，但 decoder 看不到 reference
3. **One-to-many 监督**：同样 task geometry 对应不同 reference phase 的 teacher action
4. **Motion 太少**：只有 2 条，z-space 缺乏多样性

### 8.6 Feature ABI 问题发现（2026-05-27）

> compute_task_features.py 被改过多次，导致 checkpoint 和代码的 feature 语义不匹配。

两种不同的 22D 排列（V10 原始 vs 简化版），都是 22D 所以不报错但语义完全不同。
**解决方案**：固定 V10 排列 + 4D phase = 26D，引入 `FEATURE_VERSION` 常量。

### 8.7 Prior 瓶颈深度诊断（2026-05-27）

> **核心发现**：Prior MLP 无法从 obs 推断正确的 z。问题不是 distribution shift，而是 information bottleneck + 结构不匹配。

#### 实验 1：prior_recon loss（alpha_prior=0.5）

直接在 loss 中添加 `||D(obs, P_mean(obs)) - a_teacher||^2`，训练部署路径。

| Eval | Prior Kick/Fall | Posterior Kick/Fall |
|---|---|---|
| 50 | 26% / 100% | 88% / 20% |
| **150** (best) | **66% / 50%** | 88% / 16% |
| 200 | 38% / 88% | 92% / 16% |

#### 实验 2：Prior DAgger（prior_rollout_ratio=0.3）

30% rollout step 用 prior z 执行，收集 teacher 标签在 prior 自己的状态分布上。

| Eval | PDagger Prior Kick/Fall | Baseline Prior Kick/Fall |
|---|---|---|
| 100 | 58% / 72% | 48% / 86% |
| 150 | 34% / 90% | **66% / 50%** |
| 170 | 30% / 88% | 48% / 74% |

**结论**：没有系统性改善。问题不是 distribution shift。

#### 根本原因：MLP Prior vs RNN Teacher

```
Teacher: a_t = pi(obs_t, h_t)     <- 有 LSTM hidden state
Prior:   z_t = P(obs_t)            <- 单帧 MLP

同一个 obs_t，teacher 的 hidden state 不同：
  h_approach -> 继续跑
  h_plant    -> 固定支撑脚
  h_swing    -> 加速摆腿
  h_recovery -> 收腿稳定

MLP prior 在这个 one-to-many 映射上只能取平均 -> z 不对 -> 摔倒
```

**这是结构性问题，不是训练策略问题。**

### 8.8 Stage2B-v3：RNN Prior（下一步）

#### 架构改动（只换 Prior）

```
Encoder: q(z | obs, action)       仍然 MLP
Decoder: D(obs, z)                仍然 MLP
Prior:   p(z | obs, h)            MLP -> GRU + MLP
```

GRU Prior 设计：
```
GRU(input=125D, hidden=128, num_layers=1)
  -> MLP -> (mu_p, logvar_p)

训练: 短序列 B x T，h reset at done
推理: 在线更新 h_t
```

#### 预期

| 指标 | MLP Prior (现状) | GRU Prior (预期) |
|---|---|---|
| Prior Kick% | 30-66% | 70-85%+ |
| Prior Fall% | 50-100% | 20-40% |
| Prior-Posterior gap | 30-60% | <15% |

#### 已排除

| 方案 | 原因 |
|---|---|
| Prior DAgger | 实验证明问题不是 distribution shift |
| alpha_prior loss | 在训练分布上有效但不转移到 rollout |
| z_dim=32 | prior 无法匹配，Fall%=100% |
| 更长 DAgger | 曲线先升后降，不是训练时间问题 |

### 8.9 代码文件

| 文件 | 用途 |
|---|---|
| `scripts/rsl_rl/latent_v2_models.py` | Encoder / Decoder / Prior（含 prior_recon） |
| `scripts/rsl_rl/train_latent_v2_online.py` | Stage 2B: DAgger + Prior DAgger + dual eval |
| `scripts/rsl_rl/train_latent_v2_ppo.py` | Stage 3/4: PPO + LAB |
| `scripts/rsl_rl/compute_task_features.py` | 26D task features（V10 + phase, FEATURE_VERSION） |

### 8.10 关键 Checkpoints

| Checkpoint | 说明 | Prior Kick% | Prior Fall% |
|---|---|---|---|
| `online_distill.pt` | Full obs (160D) | 96% | 18% |
| `online_distill_taskfeat.pt` | task_features (121D, 旧22D) | 61% | ~40% |
| `online_distill_v10_phase26.pt` | V10 26D + alpha_prior (best) | **66%** | **50%** |
| `online_distill_v10_phase26_pdagger.pt` | V10 26D + Prior DAgger | ~58% | ~72% |
| `model_2000.pt` (Stage 4) | PPO 99D, full-obs decoder | 82.5% Clean | 1.0% |


---

## 9. Stage C: Categorical PPO 与 Ref-Free Termination 突破

> **目标**：解决 Continuous PPO（Action Repeat）导致的 VQ Code 模糊选择问题，并彻底消除 Reference 泄漏导致的错误终止。

### 9.1 从 Gaussian PPO 到 Categorical PPO

在 V2 (Action Repeat) 实验中，使用 Continuous PPO 预测 VQ Code 的权重：
- **问题 (Code Hold/Averaging)**：由于 Gaussian 策略的连续性，模型难以做出非连续的突变决策，导致动作混杂，Fallback（补射）率高达 11%。
- **对策 (Categorical PPO)**：将 PPO 改为输出 `K=16` 的 logits（与 Markov Prior 提供的 16D prior logits 叠加 `combined = prior_logits + scale * residual_logits`），用 Categorical Distribution 进行采样。
- **成效**：Late/Fallback 率从 11% 暴降至 **3%**，动作干净果断（Clean/Attempt 达 73%）。但策略变得过于保守，No-Attempt 率高达 48%。

### 9.2 No-Attempt 诊断与 Termination 破局

为解决 Categorical PPO "不敢出脚" (No-Attempt=52%) 的问题，我们引入了基于纯空间几何的 `eval_kick_attempt_diagnostic` 工具。

**诊断发现**：
- **100% 的 No-Attempt** 都是在尚未出脚前（约 46 步，不到 1 秒）被环境强制 Terminate。
- **根因分析**：环境遗留了 `ee_body_pos` 和 `anchor_pos_z` 这些 **Reference-based Terminations**（如果真实姿态与底层的 Motion Reference 偏差 >0.25m 则判定失败）。对于一个追求自由决策（Ref-Free）的 High-Level PPO 而言，偏离参考轨迹是不可避免且被鼓励的，导致无差别误杀。

### 9.3 最终结果：禁用 Ref-based Terminations

我们在环境与评估中增加了 `--disable_ref_terminations` 选项。

| 指标 | 存在 Ref Termination | 禁用 Ref Termination |
|---|:---:|:---:|
| **Att% (尝试率)** | 48% | **100%** |
| **Clean% (干净命中)** | 45% | **94%** |
| **NoAtt% (未出脚)** | 52% | **0%** |
| Late% (补射) | 0% | 5% |
| Term% (意外倒地) | 64% | 6% |
| **Motion 4 Clean** | 54% | **100%** |

**阶段性结论**：`Ref-Free High-Level Selector + Ref-Conditioned Low-Level Decoder` 的分层框架被彻底验证有效。高层 Categorical PPO 大脑可以仅仅依靠球脚空间几何特征（task_features），在离散的 VQ 动作库中做出完美的组合决策（94% Clean Rate）。

### 9.4 泛化性验证与同口径对比 (Ball Perturbation & Old Models)

为了证明高层 Selector 不是简单地 "死记硬背" Reference Timing（固定帧起脚），我们引入了 `ball_xy_perturb` 进行球面扰动，并对被误杀的旧模型进行了**同口径重评**（去除了 ref_termination）。

**1. 泛化性极强（±0.3m 扰动几乎不掉点）**
| Ball Perturb | Clean% | Late% | NoAtt% | Att% | BSpd | DirA | M4 Clean |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ±0.0m | 94% | 5% | 0% | 100% | 7.5 | 1.00 | 100% |
| ±0.1m | 96% | 4% | 0% | 100% | 7.5 | 1.00 | 100% |
| ±0.2m | 90% | 8% | 2% | 98% | 7.5 | 1.00 | 96% |
| **±0.3m** | **93%** | **6%** | **0%** | **100%** | **7.4** | **1.00** | **99%** |
*结论：球偏了 30cm（接近一个脚掌长度），策略依然能找到球并精准踢出，证明高层确实在利用 ball-foot relation 做闭环离散选择。*

**2. 同口径去 Termination 对比 (Categorical 优势明显)**
| 方法 (去 Ref Term) | Clean% | Late% | NoAtt% | Att% | BSpd |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Categorical PPO** | **94%** | **5%** | **0%** | 100% | 7.5 |
| Gaussian V2 | 60% | 40% | 0% | 100% | 8.1 |
| VQ+Residual | 60% | 40% | 0% | 100% | 8.1 |
*结论：旧模型以前的 NoAtt=19% 完全是被环境误杀（去约束后 Att 全满）。但连续的 Gaussian/Residual 犹豫不决（Code Averaging），导致第一脚往往踢空（Late 补射高达 40%）。Categorical 离散决策完美解决了犹豫问题。*

---

## 10. 未来 Roadmap：通向完全 Deployable 的 Ref-Free 系统

虽然高层大脑（Selector）已经 Ref-Free，但底层身体（Decoder）目前仍然依赖 `MotionCommand` 提供的 phase/reference scaffold。下一步核心是清缴技术债，剥离底层依赖：

1. **固化主线 Baseline**：以 `catppo_norefterm` 作为基准标杆（Strict Clean 94%，±0.3m Perturb 93%）。
2. **One-shot Attempt 改造（处理残余的 5-13% Late）**：
   - 改革 RL Reward & Termination：Attempt 触发后只给 18 帧机会，踢中给 reward 并 success terminate，没踢中直接 failure terminate。
   - 不允许靠第二脚补射拿 reward，让训练目标与 Diagnostic strict hit 彻底一致。
3. **更系统的 Perturbation 扫雷**：拆分测前后 X 偏移、左右 Y 偏移、初始站位、目标朝向的变化，找出策略盲区。
4. **终极挑战：No-Phase / No-Reference Decoder**：
   - 重训 Decoder：`decoder(proprio/history + ball-foot features + z/code) -> action`
   - 彻底拔掉内部对 `phase`、`kick_flag` 和外部 `motion reference` 的依赖。
   - 验证：如果 No-Phase VQ 的 Post(Quant) 还能保持高表现，整个双层架构即大功告成，达到 Real-World Deployable 状态。
