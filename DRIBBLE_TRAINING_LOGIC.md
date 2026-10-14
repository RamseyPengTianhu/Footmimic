# Dribble clean-v0 训练逻辑详解

> 配套 `DRIBBLE_HANDOFF.md`(状态/数据位置/待办)。本文件解释**训练到底怎么运转**,
> 内容均来自代码阅读(2026-10-14),关键位置给出文件与类名,便于核对。
> 代码根目录:`source/whole_body_tracking/soccer/tasks/tracking/`

## 0. 一句话总览

两阶段、同一个 160 维 Actor、同一个 29 维动作:
1. **Stage1** 只学「跟着参考动作走」(motion tracking),球只是一个会出现在观测里、但无碰撞、不影响奖励的 ghost。
2. **Stage2** 从 Stage1 权重起步,在**真实有碰撞的球**上学会按参考节奏、用正确的脚和正确的面(inside/outside)触球,并把球送到离线算好的下一个节点。难度通过内部 6 级 curriculum 逐级提升。

策略不是「看到球去追球」,而是**以参考动作的时钟为主导**,球位置只作为修正量。

## 1. 数据从哪来(训练输入)

每条参考轨迹是一个 npz(50 fps)。**人工标注和球节点已经烘焙进 npz**,训练时只读 npz:

- 动作部分:`joint_pos/vel`、`body_pos_w/quat_w/lin_vel_w/ang_vel_w`(30 个 body)
- 球与触球标注:`ball_pos_w`(离线合成的球轨迹)、`dribble_cg_contact`(是否在触球窗口)、
  `dribble_cg_foot`(哪只脚)、`dribble_cg_surface`(inside/outside)、
  `dribble_contact_anchor`、`dribble_cg_foot_ball_dist`、`dribble_cg_flow_*`

生成链(都在 `scripts/dribble/`):

```
人工标注 labels.json ──► synthesize_dribble_ball_traj.py ──► 带 ball_pos_w 与 dribble_cg_* 的 npz
                                                         └─► compile_contact_nodes.py ──► *.nodes.json (+ phase_audit)
```

注意:`nodes.json` **不被训练读取**,是审计产物。训练时命令类(`commands_dribble_cg.py:645-677`)从 npz 的
`ball_pos_w` 和 contact 数组,用与 compiler 相同的公式(事件起始帧、参考骨盆 yaw 系下的球相对位置)重算节点;
对 Reference 2 的重算与 nodes.json 差 6e-8 m。世界系的触球锚点则在每个 episode 开始时按
reference→actual 的 yaw 对齐固定(`dribble_cg_episode_fixed_contact_targets=True`)。

节点坐标系:reference pelvis heading,只比较 XY 距离(`distance_dimensions: xy`)。
**节点是离线、不可变的**,不跟随机器人实时 root 移动。

## 2. Stage1:`G1CleanV0Stage1EnvCfg`

Task:`Tracking-Flat-G1-CleanV0-Stage1-RNN`,runner `G1CleanV0Stage1RecurrentPPORunnerCfg`
(RNN/LSTM PPO,`save_interval=500`)。

**观测(Actor 160 维)** = Flat 本体感觉 154 + 球在 pelvis 系下的 xyz(3,`ball_pos_pelvis_full`)+ reference twist(3,pelvis 系 `[vx, vy, wz]`)。
没有 destination。Critic 同样使用球与 twist 项。

**球的处理(`_apply_stage1_ghost_ball`,`config/g1/soccer_flat_env_cfg.py:66`)**
- 球设为 kinematic、无重力、**碰撞关闭**,不会影响机器人动力学
- 每个 episode 在 pelvis 前方工作区随机采样:`x∈[0.25,0.80]`, `y∈[-0.35,0.35]`(`soccer_ball_spawn_mode="workspace"`)
- 之后保持该 pelvis-yaw 偏移跟随实际 pelvis 平面运动(`_follow_workspace_ghost_with_actual_pelvis`),
  所以观测始终有界,不会因参考动作平移而变成几米
- 目的:让 Actor 的球观测通道在 Stage1 就「有分布」,但策略学不到依赖球的行为(没有任何奖励/终止消耗它)

**奖励(模仿为主)**
- 关节/身体姿态、朝向、相对身体位置等沿用 Flat 动作先验配方
- `motion_anchor_lin_vel`(w=2.2)、`motion_anchor_ang_vel`(w=0.4):监督的正是 Actor 看到的 pelvis-local twist
- 关闭:全局 anchor 位置/朝向奖励(学动作而非固定世界位置)、全局 body 速度项
- 保留 Flat 的 reset 噪声和周期性推力(鲁棒性)

**终止/重采样**:参考片段结束即 time-out 并重采样(`motion_clip_end_resample/terminate=True`,无 cyclic blend);另有跌倒类终止(`anchor_pos_z`、`anchor_ori`)。

产物:`2026-08-21_16-27-03_s1_clean_v0_random_bounded_ghost_1024/model_6999.pt`。

## 3. Stage2:`G1CleanV0Stage2EnvCfg`(类继承链)

```
G1FlatCGDribblingUnifiedS2LocalReferenceEnvCfg   (旧 S2 基类:接触状态机、奖励项池)
  └ G1FlatCGDribblingReferenceTwistS2EnvCfg
      └ CleanV0Stage2C2  →  T2  →  C3  →  C4  →  T4  →  CleanV0Stage2 (正式,L0–L5)
```
Runner:`G1DribblingS2RecurrentPPORunnerCfg`(init std 0.40,entropy 0.001,warm-start 重置 Adam,lr 1e-5;
注释记录:lr 1e-3 会在第一次 PPO 更新就毁掉载入的 S1 actor)。

### 3.1 观测与对称/非对称
- **Actor**:与 Stage1 完全同构 160 维(真球 xyz in pelvis + reference twist)。**看不到未来节点、不看 I3 node。**
- **Critic**:额外获得 `s2_sparse_ball_reference_context`(稀疏球目标与触球相位等特权信息),仅训练使用,部署不需要。
  检查点加载时只对新增的 critic 列补零,不动 Actor(`utils/checkpoint_loading.py`)。

### 3.2 单个事件的判定(contact event state)
由 `dribbling_s2_contact_event_state`(`mdp/rewards_dribbling.py`)统一计算,各奖励/终止项共享同一状态:

一次触球被判为**有效**需同时满足:
1. 真实 PhysX 碰撞(过滤后的 contact sensor,球对机器人所有 link;不含地面)
2. 发生在该事件的标注窗口内(窗口已含 ±2 帧标注容差;不允许窗口之前就持续接触后再声称命中)
3. 是**期望的脚**(`require_expected_foot`)
4. 是**期望的面**:inside/outside 由球心在期望脚踝 yaw-local 坐标系下的符号决定(`side_deadzone=0`,不人为加厘米余量)

失败类处理:
- 非脚部 link 碰球 → `s2_nonfoot_contact` **终止**
- 窗口过期仍未命中 → `s2_missed_contact` **终止**(clean-v0 把漏球视为普通 episode 失败,并重置参考时钟和 LSTM)
- 错脚 / 错面 / 过早接触 → 惩罚脉冲(`wrong_foot` / `wrong_side` / `premature`)

触球锚点质量(`contact_anchor_local_sigmas=(4.2,4.2,2.2)cm`,来自 264 次正确静态扫描命中的最大值)只作 shaping/遥测,**不会否决一次真实碰撞**。

### 3.3 奖励结构(clean-v0 刻意精简)
- 保留 Stage1 全套 motion prior(身体位置/朝向/速度、local twist),**脚踝位置在事件窗口前约 0.2 s 开始释放**
  (`s2_windowed_foot_tracking`,接触脚权重降到 0.3),让腿可以去够球,朝向仍被约束。
- `s2_physical_contact_candidate_bonus`(w=5.0):**唯一的**触球发现脉冲,每次有效碰撞给一次。
- **T2 起**加入球的去向奖励(只在需要的 level 评估):
  - `s2_next_anchor_progress`(w=1.0):球朝下一个固定节点的符号距离进展
  - `s2_next_anchor_window_tracking`(w=20.0):下一接触窗口结束后,用球心到**不可变 episode 固定节点**的窗口内最小距离给一次脉冲(按球半径归一化)
- 明确移除:effective/closure/settlement、point-cloud(V2/V3)、proximity、surface_style、sequence_progress、approach 系列等所有旧项(`setattr(..., None)`)。
- 所有 push 事件关闭(焦点阶段确定性,便于归因失败)。

### 3.4 内部 curriculum(单一策略、连续优化器,仅环境难度递增)

由 `mdp/curriculums.py: s2_contact_level_curriculum` 驱动,**全局且单调**(只升不降)。

| Level | 名称 | 要求的有序接触数 | 额外要求 | 晋级阈值(序列完成率) |
|---|---|---|---|---|
| L0 | C2 | 2(O1,O2) | 纯接触 | 0.80 |
| L1 | T2 | 2 | + 球在 I3 窗口内到达固定 I3 节点(终端到达) | 0.80 |
| L2 | C3 | 3 | + 真实 I3 物理触球 | 0.60 |
| L3 | C4 | 4 | 完成一个 O/O/I/I 单元 | 0.60 |
| L4 | T4 | 4 | + 到达下一周期 O1 节点 | 0.60 |
| L5 | Repeat | 8 | 两个完整 O/O/I/I 单元 | 0.50 |

晋级机制(默认参数):
- 每个评估窗口至少 1024 个 episode,**连续 2 个窗口达标**才晋级(防止单个幸运批次)
- 同时检查:接触成功率 ≥0.75、正确面率 ≥0.70、首次释放质量 ≥0.60、跌倒率 ≤0.05
- 转换类 level(L1、L4)用「终端到达成功」(基于窗口最小距离/球半径),**不是**相对改进、参考速度或实时 root 目标
- 有 plateau 检测(4 个窗口改进 <0.02)与 hard replay
- curriculum 状态会存入 checkpoint(`s2_curriculum_state`),resume 时恢复

分级 task(C2/T2/C3/C4/T4)是同一逻辑的**固定在某个 level 的调试夹具**,用于单独验证某一级;正式训练用 `Stage2-RNN` 让 curriculum 自己爬。

### 3.5 Stage2 起步方式
- 必须用 **Stage1 clean 160-D checkpoint** 作为起点(不能用含 destination 的 163-D 旧模型)
- `train_multi.py` 通过 agent 配置的 `resume / load_run / load_checkpoint` 载入(`runner.load`);
  S2 runner 配置了 warm-start 时重置 optimizer、策略 std 设为 0.40
- 具体命令行参数我没有实际跑过,首次运行需核对

## 4. 为什么这样设计(取舍)

- **reference clock 主导触球节奏**:避免策略自己决定何时踢,降低奖励设计复杂度,也方便真机用同一个时钟。
- **球节点离线固定**:旧路线用实时 root 移动目标,目标会随策略漂移,难以归因失败;现在目标不可变。
- **触球 = 物理碰撞 + 窗口 + 脚 + 面**:不依赖校准不充分的几何距离阈值,避免把真实碰撞误判为失败。
- **Stage1 放 ghost ball**:Actor 的球通道在两阶段同构,Stage2 不需要改网络输入层,也不会让动作先验学到「躲球」。
- **非对称 Actor/Critic**:未来节点等特权信息只给 Critic,保持 Actor 可部署。

## 5. 已知问题(D0 核对后)

1. ~~C4/T4 写死 O/O/I/I~~ **已核实**:`required_contacts` 是有序事件个数,不看表面类型,Reference 2 可直接用 L0–L5。
2. **phase audit**:原 30.6% 被指标缺陷夸大(intent 含 `next_event_id`);按 (脚, 表面) 重算为 17.1%,训练也不读取该文件。影响待 D1 判断。
3. **Stage2 尚无任何 clean-v0 训练 run**,Stage2 行为均来自配置阅读。
4. ~~nodes.json 一致性~~ **已核实**:训练不读 nodes.json,从 npz 用同一公式重算,与 nodes.json 差 6e-8 m。
5. 世界系触球锚点按回合起点固定,整条 clip 不重锚;持续盘带需按周期重新锚定(见 HANDOFF 第 7 节)。
6. 旧脚本已处理,见 HANDOFF D0-5。

## 6. 复现与排查路径

```bash
conda activate isaaclab_211 && source /data/isaacsim/setup_conda_env.sh
cd /home/teambruce/users/syf/projects/Footmimic

# 查看 Stage1 实际用的配置
cat logs/rsl_rl/g1_dribbling/2026-08-21_16-27-03_s1_clean_v0_random_bounded_ghost_1024/params/env.yaml
cat logs/rsl_rl/g1_dribbling/2026-08-21_16-27-03_s1_clean_v0_random_bounded_ghost_1024/params/agent.yaml

# 阅读顺序
#  config/g1/soccer_dribbling_env_cfg.py : G1CleanV0Stage1EnvCfg (l.989), G1CleanV0Stage2C2EnvCfg (l.1902) … G1CleanV0Stage2EnvCfg (l.2143)
#  mdp/rewards_dribbling.py              : dribbling_s2_contact_event_state 及 s2_* 包装函数 (~l.3752–4060)
#  mdp/curriculums.py                    : s2_contact_level_curriculum
#  mdp/commands_dribble_cg.py            : episode-fixed 节点与事件光标
#  config/g1/agents/rsl_rl_ppo_cfg.py    : G1CleanV0Stage1… / G1DribblingS2… runner
```
