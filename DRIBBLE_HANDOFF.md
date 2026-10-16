# Dribble 交接文档(clean-v0 路线)

> 本文件是盘带项目**当前状态的唯一入口**。`README_dribble.md`(旧 S1/S2/S3)、
> `S2_OUTER_INNER_REWARD_ANALYSIS.md`(旧 S2 失败分析)、`README_CN.md`(混合历史 Sprint)
> 均为历史资料,以本文件为准。
> 状态快照日期:2026-10-14。所有改动**均未提交**(分支 `ucl`,最近提交 `6fab684 6.3 new`)。

> 训练逻辑(观测、奖励、终止、curriculum 机制)详见 [DRIBBLE_TRAINING_LOGIC.md](DRIBBLE_TRAINING_LOGIC.md)。

## 1. 目标

设计**可真机部署**的干净两阶段盘带框架(Stage1 / Stage2)。

## 2. 当前正式路线

### Stage1 —— 纯 motion tracking
- Task: `Tracking-Flat-G1-CleanV0-Stage1-RNN`(`G1CleanV0Stage1EnvCfg`,
  runner `G1CleanV0Stage1RecurrentPPORunnerCfg`)
- Actor 160-D = Flat154 + 当前球 xyz(3) + reference twist(3);动作 29-D;**无 destination**
- 训练时球是随机、有界、无碰撞的 ghost ball;没有任何 reward/termination 使用它
- 不使用绝对 root 位置/朝向奖励(用 pelvis-local 的 `[vx, vy, wz]` 监督)
- **当前保留 run**:
  `logs/rsl_rl/g1_dribbling/2026-08-21_16-27-03_s1_clean_v0_random_bounded_ghost_1024`
  (7000 iter,已训完,最终 `model_6999.pt`,每 500 iter 一个 checkpoint)
- 另一个较早 run `2026-08-21_13-03-02_s1_clean_v0_local_twist_1024` 为被替换的前版

### Stage2 —— 盘带(单一内部 curriculum,策略与优化器连续)
- Task: `Tracking-Flat-G1-CleanV0-Stage2-RNN`(`G1CleanV0Stage2EnvCfg`),
  另有分级调试 task:`Stage2-C2 / T2 / C3 / C4 / T4`
  - C2:获取接触(目标为 reference 离线节点)
  - T2:到达下一个稀疏节点的 XY
  - C3:真实有序 I3 物理触球
  - C4:完成一个 O/O/I/I 单元
  - T4:到达下一周期 O1 节点
- 必须从**干净的 Stage1 checkpoint** 起步(160-D Actor 不能热启动旧的含 destination 的 163-D 模型)
- 设计约束:
  - Actor 只看可部署观测,**不能直接看到未来 I3 node**
  - reference clock 定义触球节奏
  - contact window、正确脚、inside/outside 来自**离线人工标注**
  - 球节点由 reference **离线计算**,不用实时 root 移动目标
  - missed node 视为普通 episode 结束(`rewards_dribbling.py` 内有说明)

### 已废弃(不要再用)
- 旧 V2/V3 点云、live-root target
- candidate / effective / closure / settlement 复杂体系
- 旧 `s2_*` run(2026-08-16 ~ 08-20,共约 17 个)与旧 `...-unified-s1/s2/s3-local-*` task
- `S2_OUTER_INNER_REWARD_ANALYSIS.md` 的结论

## 3. 参考轨迹与标注

| Reference | 事件数 | 脚 | 表面序列 | 标注文件 |
|---|---|---|---|---|
| `master_001675_001884` | 16 | 右 | O O I I ×4 | `scripts/dribble/strict_single_labels.json` |
| `master_003178_003326` | 8 | 右 | I O I O I O I O | `scripts/dribble/master_003178_labels.json` |

Reference 2 事件窗口(50 fps 帧):24–28 I,52–58 O,81–86 I,110–115 O,141–146 I,172–176 O,202–209 I,232–236 O。

Reference 2 的标注 → 球节点合成 → clean node 编译 → replay 视频流程已跑完,**但数据本身有竖直漂移,当前不可用于训练,见第 10 节 "V0a 几何检查"**。

## 4. 数据位置

| 内容 | 路径 |
|---|---|
| Ref 1 训练 npz | `motions/master-aligned/master_001675_001884_unitree_g1.npz` |
| Ref 2 训练 npz(含球轨迹) | `motions/master-annotated-003178-grounded/master_003178_003326_unitree_g1.npz` |
| Ref 2 无节点版 | `motions/master-annotated-003178/` |
| Ref 2 clean nodes | `motions/contact-nodes-clean-v0-003178-grounded/master_003178_003326_unitree_g1.nodes.json`(另有 `.phase_audit.json`、`summary.json`) |
| Ref 2 node replay 视频 | `logs/reference_replay/master_003178_new_nodes/master_003178_003326_unitree_g1.mp4` |
| 其他试验轨迹变体 | `motions/master-aligned-{outside-toe,toe-inside-forward*}/`(几何标定实验,非正式) |

## 5. 工具脚本(`scripts/dribble/`)

- `synthesize_dribble_ball_traj.py`:由标注合成球轨迹
- `compile_contact_nodes.py`:编译 clean-v0 节点(schema `clean_dribble_v0`,节点在 reference pelvis heading 系、XY 距离)
- `audit_contact_anchor_alignment.py` / `calibrate_foot_ball_contact.py` / `validate_*`:接触几何标定与校验
- `scripts/replay_npz_with_ball.py`:带球 replay / 出视频

## 6. 训练命令

(需 `conda activate isaaclab_211` 且 `source /data/isaacsim/setup_conda_env.sh`)

```bash
# Stage1(已完成的 run 参数:max_iterations=7000, 1024 envs)
python scripts/rsl_rl/train_multi.py \
  --task Tracking-Flat-G1-CleanV0-Stage1-RNN \
  --motion_path <motion dir> --experiment_name g1_dribbling \
  --run_name s1_clean_v0_random_bounded_ghost_1024 --num_envs 1024 --max_iterations 7000

# Stage2(从 Stage1 checkpoint 起步;resume 相关参数见 train_multi.py)
python scripts/rsl_rl/train_multi.py \
  --task Tracking-Flat-G1-CleanV0-Stage2-RNN \
  --motion_path <motion dir> --experiment_name g1_dribbling --run_name <name>
```

> 注意:`shell/progressive_dribbling_train.sh` 和 `shell/eval_dribble_references.sh`
> 仍指向**旧**的 `unified-s1/s2/s3-local-*` task,不适用于 clean-v0。上面 task 名及
> Stage2 具体 resume 参数需在首次运行时核对(本文档整理时未实际执行训练)。

## 7. 风险(D0 核对后更新,见第 10 节)

1. **phase 歧义是否影响 Stage2 尚无定论**:原始 `nearest_intent_mismatch_rate=0.306` 被指标缺陷夸大
   (见 D0-4);按 (脚, 表面) 重算为 17.1%,其中 15/48 个触球帧的最近邻意图不同,集中在窗口边缘。需要用 D1 结果判断。
2. **世界系累积偏差**:触球锚点仍按回合起点的 reference→actual 变换固定在世界系
   (`_refresh_s2_episode_fixed_targets`),整条 clip 内不重新锚定;持续盘带(D3)前必须设计按周期重新锚定。
   T2/T4 的到达判据(`relative_node_targets=True`)用的是骨盆系相对节点,不受此累积影响。
3. 所有改动已提交到 `dribble-clean-v0`;`logs/` 不在 git 里(Stage1 checkpoint 只在训练服务器上)。
4. `play_multi.py` 改动达 2500+ 行,接手时先 `git diff` 确认哪些是 clean-v0 必需。

## 8. 接手者必读清单

```bash
git status --short
git diff -- \
  source/whole_body_tracking/soccer/tasks/tracking/config/g1/soccer_dribbling_env_cfg.py \
  source/whole_body_tracking/soccer/tasks/tracking/config/g1/soccer_flat_env_cfg.py \
  source/whole_body_tracking/soccer/tasks/tracking/mdp/commands_dribble_cg.py \
  source/whole_body_tracking/soccer/tasks/tracking/mdp/commands_multi_motion_soccer.py \
  source/whole_body_tracking/soccer/tasks/tracking/mdp/rewards_dribbling.py \
  source/whole_body_tracking/soccer/tasks/tracking/mdp/curriculums.py
```

再读:第 3、4 节列出的标注、npz、nodes.json,以及 replay 视频。

## 9. 总体目标与阶段门槛(来自主线计划,docs/milestones/SOCCER_SKILLS_RESEARCH_PLAN_20261014_CN.md P2/P3)

本分支定位:训练盘带 tracker,作为将来统一 tracker 的"盘带教师"。合并时射门、盘带两个教师蒸馏出一个学生。
统一命令 = 身体速度 [vx,vy,wz] + 下一次触球期望出球速度(骨盆系向量);技能切换在 reference 层完成。

| 步骤 | 内容 | 通过条件 |
|---|---|---|
| D0 | 不训练的核对 | 每项有结论(见第 10 节) |
| D1 | C2 小探针,两条 reference 分别跑 | 有序 2 接触 ≥0.80,正确面 ≥0.70,摔倒 ≤0.05 |
| D2 | 正式课程 L0–L5 | L5 完成率 ≥0.50 |
| D3 | 循环盘带 | 命令不变时 ≥80% 回合连续触球 ≥20 次不丢球,偏差逐次收敛 |
| D4 | 速度命令 | 节奏保持下的加减速和转向 |

D3 才是真正门槛;课程是手段,达到门槛即推进,不在单条 clip 完成率上反复打磨。
D1、D2 评测需额外记录第 8、16 次触球时机器人与 reference 的位置偏差(决定"按周期重新锚定"的紧迫度)。

合并约束:球观测保持骨盆系 xyz;动作 29 维,PD/动作缩放与 G1 标准一致;不加技能 ID 观测;
新逻辑放盘带专用模块(`commands_dribble_cg.py`、`rewards_dribbling.py`),少改共用核心文件
(`commands_multi_motion_soccer.py` 等);不回到已废弃路线。
补数据:`motions/master-modified/` 还有 8 条盘带 clip 未按 clean-v0 标注,D3/D4 前标 2–4 条。球物理随机化排在 G4 前加回。

## 10. 进度日志

### D0(2026-10-14,已完成,未训练)

| # | 核对项 | 结论 |
|---|---|---|
| 1 | Stage1 可加载 | **文件层面通过**:`model_6999.pt`(iter 6999)actor 输入 160、critic 292,与 Stage2 Actor 契约一致。未在 Isaac 环境里实际 `runner.load`,该步留给 D1 首次启动。 |
| 2 | Stage2 事件表与候选 | `_build_s2_contact_event_table` 把每个连续的 `dribble_cg_contact` 段当一个事件,按时间序编号;`required_contacts=(2,2,3,4,4,8)` 是**事件个数**,与表面类型无关,表面取自 npz 每事件标注。因此 **Reference 2(I O I O…)可以直接套 L0–L5**,"O1/O2/O/O/I/I"只是 Reference 1 的命名。两条 reference 窗口在 ±2 帧容差下均无重叠。注意 Ref 2 共 8 个事件,L5(8 个)=整条 clip,Ref 1 共 16 个,L5 只覆盖前半。 |
| 3 | nodes.json 与回合固定节点 | 训练(clean-v0,`relative_node_targets=True`)在 `commands_dribble_cg.py:645-677` 用与 `compile_contact_nodes.py` **同一公式**(事件起始帧、参考骨盆 yaw 系下的球 XY/Z)从 npz 重算节点,**不读取 nodes.json**。对 Ref 2 重算与 nodes.json 最大差 6e-8 m,一致。T2/T4 到达判据是骨盆系相对量,无世界系漂移;触球锚点则在世界系按回合起点变换固定。 |
| 4 | phase_audit 是否被训练使用 | **不使用**(仅 `compile_contact_nodes.py` 写出)。且该指标有缺陷:intent 含 `next_event_id`,每个事件互不相同,导致 48/48 触球帧必然"不一致"。改为 (脚, 表面) 后 mismatch = 42/245 = 17.1%;触球帧 15/48 不一致(多在窗口边缘,如 57→116、86→29),背景帧 27/197 的最近邻落在触球上。结论:歧义存在但远小于 30.6%,是否需要 phase 通道由 D1 结果决定。 |
| 5 | 更新过时脚本 | 新增 `shell/clean_v0_dribbling_train.sh`(stage1 / stage2 两个子命令,`bash -n` 语法通过,**未执行**);`progressive_dribbling_train.sh` 标注 DEPRECATED;`eval_dribble_references.sh` 默认 task 改为 `Tracking-Flat-G1-CleanV0-Stage2-RNN`(其默认 LOAD_RUN/CHECKPOINT 仍是旧值,使用时必须显式传参)。`scripts/rsl_rl/eval_references.py` 的帮助文本仍写旧 task 名,不影响功能。 |

### D1 冒烟测试(2026-10-14,64 envs × 10 iter,Ref 1,C2 task)

- 环境:conda `isaaclab45_211`(其 `soccer` 包指向本仓库;`isaaclab45_211_ablation` 指向 `projects/ab/Footmimic`,不要混用)。
  文档里早先写的 `isaaclab_211` 和 `/data/isaacsim/setup_conda_env.sh` 在本机不存在,Isaac Sim 为 pip 安装,无需 source。
- **发现并修复 1 个 bug**:`commands_dribble_cg.py::_set_debug_vis_impl` 在 `class_type` 被换成盘带命令、而 cfg 仍是
  `MotionCommandCfg` 时访问不存在的 `s2_fixed_target_marker_cfg`,环境创建即崩。已加保护(没有 S2 marker cfg 就不建 S2 标记)。
- 修复后:Stage1 `model_6999.pt` 成功加载(日志:obs/action 接口自动适配、obs normalizer 扩展、S2 optimizer 与策略 std 重置、
  curriculum 从 level 0 开始);10 个 iteration 正常完成,无 NaN/崩溃。第 10 轮 mean reward 4.34,mean episode length 43 步,
  终止几乎全部是 `s2_missed_contact`(符合从未接触过球的起点预期);contact_success_rate 为 0,10 轮不足以判断。
- 注意:resume 后 iteration 计数从 7000 继续,`--max_iterations N` 表示再训练 N 轮。
- 冒烟 run 目录:`logs/rsl_rl/g1_dribbling/2026-10-14_15-18-31_d1_smoke_ref1`(可删除)。

### 训练逻辑审计(2026-10-14,D1 进行中时做的,基于 env.yaml 与参考数据)

结论:模仿 + 不可变节点 + 物理触球的结构自洽,但**球这部分的训练信号太弱且是离散的**,与"最大问题"的判断一致。

事实(均已核对):
- Actor 观测 = 参考命令(关节 pos/vel)+ 重力 + 角速度 + 关节状态 + 上一动作 + **球 xyz(骨盆系,仅位置)** + 参考 twist。
  **没有球速度、没有期望球位置/速度**;触球时机只通过参考时钟(命令里的相位)隐式获得。接触图(contact graph)只存在于奖励和 Critic 特权观测里。
- C2 的奖励里与球有关的只有:`s2_physical_contact_candidate_bonus`(w=5,一次碰撞脉冲)和三个脉冲惩罚(错脚 -1、错面 -1.5、过早 -1)。
  所有 approach / proximity / region / 球轨迹类连续项都被置 None。其余全部是人体模仿项。T2 才加入 `next_anchor_progress`(w=1)与窗口到达脉冲(w=20),且只在首次触球之后。
- 触球要求精度很高(`contact_anchor_local_sigmas = 4.2/4.2/2.2 cm`),而 `motion_body_pos` 的 std 是 30 cm,窗口内接触脚的位置跟踪还被降到 0.3 倍。
- 参考球轨迹**是合成的**:`synthesize_dribble_ball_traj.py` 在人工标注的触球锚点之间做匀速线性插值(z 恒为 0.11),没有真实球运动数据;
  该脚本自己的文档也写明 Stage2 用事件标签,"而不是把插值路径当作 ground truth"。因此不应该直接对整段合成球轨迹做稠密跟踪。
- 指标里没有"窗口内脚到球的最小距离",所以无法区分是差 3 cm 还是差 30 cm。

D1 现象:C2 起点所有 episode 恰好在第 43 步(第一个窗口 +grace 过期)因 `s2_missed_contact` 结束,碰撞奖励一直为 0。
原因尚未证实,最可能是"仅有二值碰撞信号 + 厘米级精度要求"(假设,待诊断)。

候选改动(均未实施,需决定):
1. 先做无训练诊断:冻结 Stage1 策略,记录每个窗口的脚-球最小距离。
2. 用**参考导出的**脚-球相对向量(踝关节局部系)在窗口前后做稠密奖励(人-物交互模仿式),替代被关闭的 approach 项;不是实时 root 目标。
3. 用连续的"期望出球速度"(由相邻节点/时间差算出,骨盆系)奖励实际出球速度;这与主线统一命令的"出球速度向量"是同一量,利于合并。
4. Actor 加球速度观测(3 维)或依赖 LSTM 推断;这会偏离"球观测保持骨盆系 xyz"的合并约束,需与主线协商。

### V0a 几何检查(2026-10-14,不训练)

用 `scripts/dribble/calibrate_foot_ball_contact.py`(机器人固定在参考姿态,球放在锚点周围 5×5×3 网格,±2 帧时间偏移,只认脚的过滤接触)检查两条 reference。
(`validate_dynamic_reference_contacts.py` 的开环 PD 回放 0/6 命中,但它不带平衡,0.3 s 内脚位置误差就到 0.33 m,不能用来判断几何,结论不采用。)

**Reference 1(grounded,最低脚高约 4–5 cm)**
- 锚点在踝局部系里很一致:inside [8.5, 14.0, 3.1] cm,outside [7.7, -12.5, 5.3] cm,标准差 1–2.4 cm;球心到踝约 16 cm。
- 但名义球位置(恰好锚点)在时间偏移 0 的帧上只有 1/16 个事件命中;多数事件在 +1/+2 帧或 -2/-1 帧才命中,4/16 个事件在整个 ±2 帧、±4 cm 网格内的任何组合下都不命中率极低(整体网格命中率 33%)。
  含义:标注帧和实际接触帧有 1–2 帧偏差;静止切线球对厘米级几何极敏感。
- 注:Reference 1 的 outside 锚点与合成脚本默认参数相差 5.6–6 cm(inside 为 0),说明 `master-aligned` 是用非默认 outside 参数合成的,参数未记录。`audit_contact_anchor_alignment.py` 已修复缺参数问题,但它对 Reference 1 的 outside 事件会报 FAIL,原因就是这个。

**Reference 2(`motions/master-annotated-003178*`)——有竖直漂移,不可用**
- 骨盆高度 82 → 103 cm,最低脚高 9 → 27 cm(每 30 帧采样),Reference 1 稳定在 4–5 cm。后半段机器人是"飘着"的。
- 标注帧的右踝高度 16, 24, 28, 32, 31, 36, 42, 32 cm(球心 11 cm),脚根本够不到地面的球;锚点在踝局部系中离脚 23–34 cm(Ref 1 约 16 cm)。
- 静态测试整体命中率 8.4%,事件 3–7 在整个网格内命中为 0。
- 合成锚点与生产几何一致(误差 0),仿真与 npz 的踝位置只差 0.2 cm:问题在**动作数据的竖直漂移**,不是合成/仿真流程错误。
- 修复路径(未做):`scripts/dribble/ground_motion_ref.py`(每帧让最低脚底回到地面,整体 Z 平移,水平不变)→ 用 Reference 1 同样的合成参数重新合成球 → 重编译 nodes → 重跑静态测试 → 重新核对标注帧(漂移被去掉后,标注帧是否对应脚真的靠近球)。
- **已修复(2026-10-14)**:新增 `motions/master-annotated-003178-grounded/`(`ground_motion_ref.py` 接地 + 用 Reference 1 的合成参数重新合成)与
  `motions/contact-nodes-clean-v0-003178-grounded/`(重新编译的 nodes)。原 `master-annotated-003178*` 目录**废弃,不要再用**。
  - 接地后:骨盆 78–82 cm、最低脚 4–5 cm(与 Reference 1 一致);标注帧右踝 12–15 cm(仅 ev6 为 23 cm),踝-球 3D 距离 14–20 cm(Reference 1 为 15–18 cm)。
  - 静态测试:网格命中率 31.9%(Reference 1 为 32.8%,漂移版为 8.4%);名义球位置在 ±2 帧内命中的事件 7/8(Reference 1 为 12/16)。
  - 节点 z 稳定在 -0.64…-0.72(漂移版漂到 -0.91)。
  - 仍有的规律:inside 事件的名义球多在 +1/+2 帧才命中(标注偏早 1–2 帧),outside 事件多在 -1 帧命中(标注偏晚 1 帧);Reference 1 类似。
- **Reference 1 的合成参数(反推,锚点误差 0)**:`--outside_foot_contact_x 0.10`,其余默认(`inside_foot_contact_x 0.10`、`outside_lateral_inset 0.0`)。
  因此审计脚本需带 `--outside_foot_contact_x 0.10` 才会对 Reference 1 PASS。
- `phase_audit` 的 30.6% 在接地后不变(它只看关节/twist,与高度无关),该指标缺陷见 D0-4。

### V0b:冻结 Stage1 策略探针(2026-10-14,不训练,Reference 1,`scripts/dribble/probe_policy_foot_ball.py`)

探针加载 Stage1 `model_6999.pt`,在 C2 环境里跑整条 clip(去掉漏接触/非脚接触/球丢失终止),按逐步时间序列记录。
注:探针里的接触传感器读数不可信(球已被踢动的环境传感器仍为 0),**触球以球速度 >0.1 m/s 判定**。

| 结果 | 数据 |
|---|---|
| Stage1 自己的环境(幽灵球) | 64/64 走完整条 clip(330/331 帧),0 摔倒 |
| C2 环境(真实球,球留在原地) | 32/32 在第 129–204 帧因 `anchor_pos_z` 摔倒,中位 140;摔倒时球与机器人距离中位 2.7 m(p10–p90 2.1–3.6 m),与噪声无关 |
| **因果测试**:C2 环境但每步把球放回骨盆前 0.5 m | **64/64 走完整条 clip,0 摔倒** |
| 动作噪声 0 / 0.2 / 0.4 下"球被踢动"的环境比例 | 31% / 62% / 73%(噪声没有毁掉触球,反而增加) |
| 第一次踢动发生的位置 | 主要在 ev1 窗口(帧 45–53),在 ev0 窗口(33–42)内只有 3% / 14% / 8%;另有 6–12% 在 ev0 窗口之前(帧 30–32,过早) |
| 没踢到的环境最近距离 | ankle–球 18–21 cm(能踢到的是 14–19 cm),只差几厘米 |

结论:
1. **多次触球学不出来的直接机制**:第一脚之后球不再跟着机器人,球在骨盆系中的位置变成 Stage1 从没见过的值(Stage1 训练时球始终在身前 0.25–0.8 m 内),策略 130–200 帧内崩溃。这与奖励设计无关,是**观测分布**问题。
2. 所以"保持球在身边"本身就是多次触球的必要条件,需要一个贯穿全程的稠密信号,而不是一个个孤立的触球事件。
3. 旧训练的 0 有效接触并非因为探索噪声(见上),更可能是:第一个事件(ev0)窗口内的触球本来就少(3–14%),而旧 C2 在 ev0 窗口关闭时就终止 episode,有效触球要同时满足窗口/脚/面,且不允许提前接触。
4. Stage1 训练用的是 `master-aligned-toe-inside-forward` 变体,与 `master-aligned` 的身体动作逐元素相同,仅球位置差最大 6 cm,不影响 Stage1 先验。

### 下一步:D1
- 用 `shell/clean_v0_dribbling_train.sh stage2` 以 `model_6999.pt` 起步,先用 `Tracking-Flat-G1-CleanV0-Stage2-C2-RNN`(设 `S2_TASK`)小规模探针,两条 reference 各一次
- 记录:run 名、checkpoint、有序 2 接触成功率、正确面率、摔倒率、第 8/16 次触球偏差(C2 仅 2 接触,第 8/16 次偏差需在 D2 才有)
- 风险提示:Stage2 的 warm-start 路径(`checkpoint_loading.py`、critic 292→含 S2 上下文的补零)从未在 clean-v0 上实跑


## 11. 主线决定与架构无关工作(2026-10-15)

**主线决定**:控制器改为"reference → encoder → latent → tracker",运行时不再输入完整 motion reference。盘带 = 命令驱动、由人体节奏组织;
命令 = 身体速度(Actor 已观测的 reference 速度);球奖励按命令定义(球保持在身前、球速跟随命令);离线节点只作诊断,不进奖励;
人体数据和 Contact Graph 负责何时、哪只脚、哪个面触球,clean-v0 的触球判定全部保留。盘带将直接在 latent 架构里训练(Stage 1′ latent 动作跟踪,
Stage 2′ RL 学带球),**等主线在射门上验证完再开始**。在此之前只做架构无关的工作,**不推倒 clean-v0**。

### 本分支状态更正
- 本分支做过一次 V1-a 实验(骨盆系球位置跟踪,300 轮),**负面结果**:奖励被取巧(机器人放慢脚步守着静止球),见 `DRIBBLE_MINIMAL_PLAN.md`。
  原因是目标随机器人骨盆移动(live-root 目标),不是 clean-v0 的设计。该任务 `Tracking-Flat-G1-CleanV1-BallTrack-RNN` 仍在代码里,但**不要使用**。
- 评估工具 `probe_policy_foot_ball.py` + `eval_dribble_metrics.py` 可复用(走完 clip 比例、触球次数、球丢失比例)。

### 四项工作
**1. 球的滚动阻力** —— 辅助函数 `apply_ball_damping(scene_cfg, linear_damping, angular_damping)`(`soccer_flat_env_cfg.py`),默认不改变现有行为。
(必须是辅助函数,不能做成 scene 配置字段:`InteractiveScene` 把每个字段当资产。)测量脚本 `scripts/dribble/measure_ball_roll.py`。
踢出 1.5 m/s 的球,测得(速度均为测量值;踢出后约 0.1 s 内由滑动变滚动,速度降到 5/7):

| 线性阻尼 | 角阻尼 | 0.5 s | 3 s | 12 s | 12 s 保留 | 1 s 处减速度 | 3 s 滚动距离 | 12 s 距离 | 停下 |
|---|---|---|---|---|---|---|---|---|---|
| 无 | 无 | 1.06 | 1.03 | 0.90 | 84% | 0.015 m/s² | 3.14 m | 11.8 m | 不停 |
| 0.05 | — | 1.04 | 0.92 | 0.59 | 56% | 0.051 | 2.97 | 9.6 | 不停 |
| 0.1 | — | 1.03 | 0.83 | 0.38 | 37% | 0.085 | 2.82 | 8.0 | 不停 |
| 0.2 | — | 0.99 | 0.67 | 0.16 | 17% | 0.145 | 2.54 | 5.8 | 不停 |
| 0.4 | — | 0.92 | 0.43 | 0.03 | 3% | 0.239 | 2.09 | 3.4 | 10.2 s |
| — | 0.5 | 1.00 | 0.70 | 0.20 | 20% | 0.133 | 2.62 | 6.2 | 不停 |
| — | 2 | 0.83 | 0.20 | 0.00 | 0% | 0.361 | 1.57 | 1.9 | 5.4 s |
| — | 5 | 0.56 | 0.02 | 0.00 | 0% | 0.429 | 0.78 | 0.8 | 2.2 s |

- 无阻尼基线与主线观察一致(12 s 约降 16%)。
- **线性阻尼更合适**:角阻尼会同时吃掉踢出后的初速度(角阻尼 2 时 0.5 s 已降到 0.83,5 时降到 0.56),扭曲踢球响应;线性阻尼基本保留初速度。
- **未选定最终值**——参数必须与主线一致。需要主线给出目标:想让球在多少秒内停、或某速度下的减速度(室内硬地约 0.1–0.3 m/s²,草地/人造草皮更大,这是量级估计,非实测)。
  线性阻尼 0.2 ≈ 0.15 m/s²,0.4 ≈ 0.24 m/s²;要 0.5 m/s² 量级需要约 0.8–1.0(未测)。定下后同步。

**2. 两个球奖励(已写好并测试,未接入任何训练任务)** —— `mdp/ball_reward_math.py`(纯 torch)+ `rewards_dribbling.py` 里的
`dribbling_ball_front_zone`、`dribbling_ball_velocity_follow`。
- 球在身前区间:骨盆 yaw 系,x∈[0.15,0.70]、|y|≤0.40,区间内为 1,区间外高斯衰减(softness 0.25)。
- 球速跟随命令:球速度(骨盆 yaw 系)与命令 `[vx,vy]`(`locomotion_twist_command_b()`,Actor 已观测的同一个量)的向量误差,`exp(−err/0.5)`,长程梯度。
- 测试:`scripts/dribble/test_ball_rewards.py`(16 项,全过);离线 `offline_ball_reward_check.py`:只看指令速度 >0.8 m/s 的行走阶段,
  参考上界 区间×速度=0.46;失败的 V1-a 取巧策略(静止球)0.02–0.03,冻结 Stage1 0.03,即约 15–20 倍区分度。
- 注意:参考开头约 1 s 人本来走得很慢(0.25 m/s),若包含这段,静止球也能拿到较高的速度跟随分(0.25),离线评估必须排除。
- 参考自身的速度跟随只有约 0.46(合成球路径是锚点间匀速,与逐帧身体速度有差),所以该项的可达上限约 0.5,不是 1。
- 已知取巧空间(未处理):用小腿持续顶着球走(球速≈身体速度)能同时拿满两项;现有 `undesired_contacts` 惩罚非脚接触,是否足够待训练时观察。

**3. 接触时钟** —— `scripts/dribble/extract_contact_clock.py`,输出 `motions/contact-clock/*.contact_clock.json`(事件、窗口±2 帧、脚、面、间隔、重复单元、逐帧下一次触球的剩余帧数/脚/面)。
| clip | 事件 | 间隔 | 触球频率 | 重复单元 |
|---|---|---|---|---|
| master_001675_001884 | 16(全右脚) | 19.3±3.6 帧(0.39 s) | 2.76/s | 出-出-内-内 |
| master_003178_003326(接地版) | 8(全右脚) | 29.7±1.0 帧(0.59 s) | 1.92/s | 内-出 |
- **主线的 `scripts/motion_gen/contact_anchors.py` 在本仓库任何分支/磁盘上都找不到**(upstream 只有 main、dribble、dribble-clean-v0),所以没有复用;
  上面的 schema 刻意做小,便于映射。需要主线把该脚本推过来,或说明它的接口。

**4. 数据整理** —— 盘点脚本 `scripts/dribble/inventory_motions.py`(长度、速度、转角、骨盆/脚竖直漂移、标注状态)。
- **`motions/master`、`master-modified` 里 10 条盘带 clip 中 9 条有竖直漂移**(最低脚高到 19–56 cm,与 Reference 2 同一问题),只有 `master_003560_003680` 没有。**标注前必须先接地。**
- 已接地 4 条候选放进 `motions/dribble-candidates-grounded/`(接地后骨盆 70–80 cm、最低脚 4 cm),**球和触球标注尚未重做**:这些文件里的触球标签来自旧流程,不是 clean-v0 标准,需要人工标注后用 `synthesize_dribble_ball_traj.py --outside_foot_contact_x 0.10` 重合成球。
  | clip | 帧数 | 速度 | 转角 | 备注 |
  |---|---|---|---|---|
  | master_003560_003680 | 199 | 1.40 m/s | 36° | 原本无漂移,有转向 |
  | master_002200_002367 | 277 | 1.10 m/s | 42° | 有转向 |
  | master_002640_002861 | 367 | 0.59 m/s | 6° | 慢速 |
  | master_006573_006727 | 255 | 0.58 m/s | 2° | 慢速 |
- 另有 `motions/dribble/`(FAST-seg*、1d1l-seg*,437–1397 帧,0.3–1.5 m/s,有大转角,如 1d1l-seg3 转 88°)是**无触球标注的连续带球段**,对"按命令转向"有价值,但需要从零标注。
- **行走、站立数据:本分支没有专门的行走/站立数据集**。最接近的是 `soccer-standard`(10 条)与 `soccer-stylized`(3 条),都是低速接近 + 踢球(0.18–0.37 m/s),以及 `dribble/FAST-seg*` 的慢走段(0.3–0.5 m/s)。如果主线有专门的行走/站立数据,需要说明位置。
- 标注本身(哪一帧哪只脚哪个面)是人工工作,我做不了。

### 需要主线/你确认
1. 球阻尼的目标值(见上表),定下后我把它加进盘带任务的配置。
2. `contact_anchors.py` 的位置或接口。
3. 行走/站立数据在哪里。
4. 补标的 2–4 条:建议先标 `master_003560_003680`(无漂移、有转向)和 `master_002200_002367`(转向),再选一条慢速。


## 12. 主线回复后的更新(2026-10-15,稍晚)

`0a7cf51` 已推送到 `upstream/dribble-clean-v0`。以下为其后的新工作,未推送部分见本节末尾的提交说明。

### 12.1 恒定减速度的球阻力(主线目标:μr=0.05,即 0.491 m/s²,不用线性阻尼)
实现(`mdp/ball_physics.py`):
- `apply_ball_rolling_resistance(ball, mu_r, ..., force_scale)`:按规格,水平力 μ·m·g、方向与水平速度相反、低于 0.05 m/s 不施加;额外有 `m*speed/hold_dt` 上限防止一个控制步内反向。
- `apply_ball_velocity_level_deceleration(ball, decel, v_min, dt)`:**引擎无关版本**,每个物理步直接把水平速度减去 decel·dt,同时按同一比例缩水平角速度(滚动球保持纯滚动)。
- 两个事件包装:`ball_rolling_resistance_event`、`ball_constant_deceleration_event`(每控制步一次)。
- `apply_ball_damping` 保留(主线要求),`apply_ball_damping(scene_cfg, 0.0, 0.0)` 用来把资产自带阻尼清零。

实测(`measure_ball_roll.py`,滚动起始,球已按纯滚动给初速度,12–20 s;机器人停在 500 m 外以免撞球):

| 实现 | 1.5 m/s 减速度 | 5 m/s | 9 m/s | 1.5 m/s 停止 | 5 m/s 停止 | 9 m/s 停止 |
|---|---|---|---|---|---|---|
| 规格目标 μg | 0.491 | 0.491 | 0.491 | 3.1 s / 2.3 m | 10.2 s / 25.5 m | 18.3 s / 82.6 m |
| 力 μ·m·g,资产默认阻尼 | 0.366 | 0.414 | 0.470 | 4.0 s / 3.0 m | 12.8 s / 31.3 m | 超 14 s / >90 m |
| 力 μ·m·g,阻尼=0 | **0.350** | **0.350** | **0.350** | 4.1 s / 3.2 m | 14.1 s / 35.6 m | — / 109.7 m |
| **力 1.4×μ·m·g,阻尼=0** | **0.491** | **0.491** | **0.491** | **2.9 s / 2.3 m** | **10.1 s / 25.4 m** | **18.2 s / 82.4 m** |
| 力 1.4×μ·m·g,资产默认阻尼 | 0.503 | 0.552 | 0.608 | 2.9 s / 2.2 m | 9.4 s / 23.1 m | 16.2 s / 70.3 m |
| 速度层面 0.4905,资产默认阻尼 | 与"力 1.4×,默认阻尼"逐点相同(最大差 <1 mm/s) | | | | | |

结论与对主线的提示:
1. **按规格原样的力(μ·m·g)只给出 0.350 m/s²,比目标低 29%。** 原因是滚动球的质心力要和转动共担:有效减速度 = F/(m(1+I/(mR²)))。Isaac 里的球是实心球(I=0.4 mR²),因子正好 1.4,实测 0.350 = 0.491/1.4。
2. **这个因子是引擎相关的**:MuJoCo 里球的转动惯量若不同,同样的力会给出不同的减速度。"两边用同一个数值"用力的方式做不到,除非两边都标定一个 force_scale。**建议用速度层面版本**:它与 Isaac 的"力 ×1.4"完全等价(已验证),并且不依赖转动惯量。
3. **必须把资产自带的线性/角阻尼显式置 0**:否则残余阻尼叠加一个与速度成正比的减速度(+0.06 m/s² @5 m/s、+0.12 @9 m/s,约 0.013·v),恒定减速度就不恒定了。置 0 后减速度在 1.5/5/9 m/s 全程精确恒定,停止时间距离与规格吻合。
4. 控制步(0.02 s)更新与物理步(0.005 s)更新在力版本下结果相同(1.5/5/9 m/s 逐点一致)。
5. 低于 0.05 m/s 不施加,所以球会以 ≤0.05 m/s 缓慢爬行而不是严格停下(按规格)。
6. 这些规格只在 Isaac 验证过;滑动起始(未滚动的踢球)有 ~0.1 s 的滑动→滚动过渡,速度降到 5/7(见 11 节),与阻力无关。

**未接入任何训练任务**:`ball_constant_deceleration_event` 需要作为 interval 事件(间隔=控制步)加进环境的 `events`,并配合 `apply_ball_damping(scene_cfg, 0.0, 0.0)`。因为盘带将转入 latent 架构,这一步留到主线定下接口后再做。

### 12.2 `contact_anchors`(按主线给的接口)
`scripts/dribble/contact_anchors.py`:`ContactAnchor` 数据类与 `load_contact_anchors(npz_path)` 与主线接口一致(盘带按 `dribble_cg_contact` 连续段切分、`dribble_cg_foot`=1 为右脚、方向/距离/时长取段内第一个 `dribble_cg_flow_valid` 帧;踢球 clip 一个锚点,方向与力度留空)。
测试 `scripts/dribble/test_contact_anchors.py`:两条参考共 **24 个锚点,全部右脚,帧号与表面序列一致**(与主线的测试结论相同)。流向字段在每条 clip 里少一个(最后一次触球没有下一个目标),符合预期。
文件放在 `scripts/dribble/` 下,未占用主线的 `scripts/motion_gen/contact_anchors.py` 路径,避免合并冲突;主线那份提交后可直接替换。

### 12.3 速度跟随的"按触球周期"版本——**上限没有接近 1**
主线建议:两次触球之间,球的平均速度(位移/时间)对比同一段内的平均命令速度,预期上限接近 1。离线实测(`offline_cycle_velocity_ceiling.py`,参考自身):

| 参考 | 区间数 | 球平均速率 | 命令平均速率 | 速度误差均值 | 周期平均得分(std 0.5) |
|---|---|---|---|---|---|
| Reference 1 | 15 | 1.33 m/s | 1.26 m/s | 0.46 m/s | **0.44**(0.18–0.80) |
| Reference 2(接地) | 7 | 1.93 m/s | 1.80 m/s | 0.39 m/s | **0.47**(0.33–0.61) |

与瞬时版本(0.43–0.47)几乎没有区别,所以**周期平均并没有解决上限问题**。把误差拆成两个分量:

| 参考 | 沿指令方向误差 | **横向误差** | 只看沿指令分量的得分 | 全向量 std 1.0 的得分 |
|---|---|---|---|---|
| Reference 1 | 0.18 m/s | **0.37 m/s** | **0.74** | 0.65 |
| Reference 2 | 0.14 m/s | **0.34 m/s** | **0.76** | 0.68 |

**瓶颈是横向分量**:盘带是内、外侧交替触球,球本来就在身体两侧来回,平均横向速度对不上身体速度。这是盘带的结构特征,不是噪声,也不会因为平均而消失。
建议:速度跟随拆成**沿指令方向(紧,std≈0.5)+ 横向(松,std≈1.0 或只在球离开身前区间时才惩罚)**,横向由"球在身前区间"那一项管。这样上限约 0.75。
**加了滚动阻力之后参考的这些分数会变,需要重测**(参考球路径是合成的匀速,没有阻力)。

### 12.4 行走/站立数据与补标
- 行走/站立数据在主线机器上、不在 git:`s833_executable_amass_stand_ready_npz`(站立 8 条约 86 s)、`s847_amass_walk_probe_npz`(行走 3 条约 17 s)。**向前 1.0–1.8 m/s 常速行走基本空缺**,主线建议用 s830 同一转换流程补 10–20 条向前行走/慢跑。拷贝方式需要你决定(scp 或提交到某分支)。
- `contact_anchors.py`(主线版)同样不在 git,需要你决定 scp 还是推送。
- 补标顺序同意:先 `master_003560_003680`、`master_002200_002367`(有转向),再选一条慢速。标注本身需要人工。


## 13. 主线补充后的更新(2026-10-15,读取 656b6d7 之后)

### 13.1 阻力:低速归零 + 贴地条件(已实现并验证)
`mdp/ball_physics.py` 的 `apply_ball_velocity_level_deceleration` 和 `apply_ball_rolling_resistance` 现在:
- **低速归零**:水平速度 ≤ 0.05 m/s 时,把水平速度和水平方向的角速度直接置零(竖直速度和绕竖直轴的自旋不动)。
- **贴地条件**:球心高度 ≤ 球半径 + 2 cm(相对 `ground_z`,事件包装里取 `env_origins[:,2]`)才施加;腾空不减速。
- 与资产阻尼清零(`apply_ball_damping(scene_cfg, 0.0, 0.0)`)配合使用。

验证(阻尼=0,滚动起始,`measure_ball_roll.py`,速度层面版本与"力 1.4×"两种实现结果一致):
| 初速度 | 有效减速度 | 速度精确为 0 的时刻(规格) | 停下后最后 1 s 最大速度 |
|---|---|---|---|
| 1.5 m/s | 0.491 m/s² | 2.98 s(3.1 s) | 0.0000 |
| 5 m/s | 0.490 | 10.10 s(10.2 s) | 0.0000 |
| 9 m/s | 0.491 | 18.26 s(18.3 s) | 0.0000 |
- 球不再蠕动,速度跳到 0 之前最后一个非零值是 0.048–0.055 m/s(即阈值附近)。
- 腾空测试(vx=3 m/s、vz=2.5 m/s 向上):飞行 0.6 s 内水平速度保持 3.00 不变(与无阻力对照一致);落地后恢复减速(1.0 s 时 2.75,1.8 s 时 2.36,即约 0.49 m/s²)。
- 注意:低速归零每步都执行,球被脚轻推时若瞬时水平速度低于 0.05 m/s 会被清零一步;接触后求解器通常立刻给出更大的速度,影响应可忽略,训练接入时需观察。
- **未接入任何训练任务**(按主线要求,等 latent 架构接口定下再接)。

### 13.2 速度跟随按"完整触球面周期"重测——**上限恢复到 0.78–0.89**
`scripts/dribble/offline_cycle_velocity_ceiling.py` 现在比较第 k 次到第 k+L 次触球之间球的平均速度与同一段内平均命令速度(L = 一个完整触球面周期)。

| 参考 | 窗口 | n | 球速率 | 命令速率 | 误差(全/沿/横) m/s | 全向量得分 | 沿指令(0.5) | 横向(1.0) |
|---|---|---|---|---|---|---|---|---|
| Reference 1(出-出-内-内) | 单次间隔 L=1 | 15 | 1.33 | 1.26 | 0.46 / 0.18 / 0.37 | 0.44 | 0.74 | 0.71 |
| | **完整周期 L=4** | 12 | 1.19 | 1.18 | **0.06 / 0.02 / 0.05** | **0.89** | 0.95 | 0.95 |
| Reference 2 接地(内-出) | 单次间隔 L=1 | 7 | 1.93 | 1.80 | 0.39 / 0.14 / 0.34 | 0.47 | 0.76 | 0.72 |
| | **完整周期 L=2** | 6 | 1.69 | 1.73 | **0.12 / 0.10 / 0.05** | **0.78** | 0.82 | 0.95 |
- 主线的判断是对的:上一节(12.3)我只测了单次触球间隔(0.44–0.47),完整周期里左右偏移互相抵消,横向误差只剩 0.05 m/s。
- 周期平均的全向量得分已够高(0.78–0.89),**暂不需要拆成沿指令/横向两项**;拆分后的分数(沿指令 0.82–0.95、横向 0.95)作为备选保留。
- 仍需注意:参考球路径是合成的匀速、没有阻力,加入滚动阻力后这些分数要重测;窗口数量少(6–12),方差未评估。
- 12.3 节中"建议拆成沿指令+横向"的结论已被本节取代。


## 14. 架构无关工作的补完(2026-10-16)

### 14.1 周期速度跟随的有状态奖励
`mdp/ball_reward_math.py::CycleVelocityFollow`(纯 torch)+ `rewards_dribbling.py::dribbling_cycle_velocity_follow`(薄封装)。
- 周期 = 一个完整触球面周期(Reference 1:4 次,Reference 2:2 次),**边界来自参考接触时钟**,不依赖检测到的物理触球,漏触不会让信号消失。
- 周期内累计命令世界速度,到下一个边界时比较球的平均速度与平均命令速度,在边界步给出脉冲 `exp(−|差|/0.5)`,其余步为 0。首个边界只开始周期,不给分。
- 测试 `scripts/dribble/test_cycle_velocity_follow.py`(10 项全过):逐帧回放参考,周期脉冲均值 Reference 1 为 0.92(离线重叠窗口版 0.89)、Reference 2 为 0.84(0.78)——差别来自这里只有 3/1 个互不重叠的完整周期;
  静止球 0.11、球逆命令方向 0.01、命令是球速两倍 0.11,均被压低。
- **Isaac 冒烟测试(2026-10-16,已做)**:`probe_policy_foot_ball.py --cycle_reward_len 4`,C2 环境、Reference 1、16 个环境、球每步放回身前(让策略能走完整条 clip),整条 clip 走完。结果:脉冲只出现在周期边界帧 107、185、263(16/16 个环境),**首个边界 35 没有脉冲**(只开始周期),其余步为 0;脉冲值 0.55–0.97,中位 0.84。球跟着机器人走,所以分数高是预期的,**这个数值不代表策略学会了带球**。封装与参考接触时钟对得上。
  - **跨 reset 已验证(2026-10-16)**:`--no_early_stop --max_steps 720`,每个 episode 里脉冲都只出现在 107/185/263,首个边界 35 从不给分,reset 后行为正确;出现脉冲的 episode 三个脉冲齐全。部分环境后面的 episode 较短(推测提前摔倒,未追查)。
- 这是**脉冲**奖励(每周期一次),不是稠密信号;与"球在身前区间"(稠密)配合使用。是否需要稠密版本取决于训练里的表现,没有验证。
- 未接入任何训练任务(按主线要求)。

### 14.2 D3 验收脚本
`scripts/dribble/eval_d3_acceptance.py <probe.npz>`(合成数据测试 `test_d3_acceptance.py` 通过):
- `gate`:≥N 次触球(默认 20)且未丢球/未摔倒的回合比例(主线门槛 ≥0.80);触球次数分布;
- 球在身前区间的时间占比;第 8、16 次触球时机器人位置对对齐参考的偏差、球在骨盆系里对参考的偏差(参考按前 1.2 s 做刚体对齐);
- `carry_suspect`:取巧检测(球持续贴着踝关节且速度 >0.3 m/s 超过 0.5 s)。**几何启发式,会误报**;探针的接触传感器读数不可信,所以做不了按身体部位的判断。
- 局限:探针只记录右踝(两条参考都是右脚);参考 clip 只有 8–16 次触球,**N=20 在不循环 clip 的情况下无法达到**。

当前基线(冻结 Stage1 + 真实球,门槛暂降到 1 次触球仅用于看输出):门槛通过率 0.00–0.02,平均触球 0.2–1.1 次,球在身前区间占比 0.03–0.17。这是起点,不是结果。

### 14.3 远程会话
服务器 tmux 会话 `claude-rc` 里有一个从本对话 fork 出的 Claude Code 会话(`/rc active`),电脑关机后仍在服务器上运行。脚本 `~/.local/bin/claude-rc`(不在仓库里)可另起一个新的远程会话。训练/测量用 `nohup` 起的后台脚本不依赖会话,但监控通知依赖会话,会话断开后要手动读日志。

### 14.4 仍在等主线 / 你
- 推送:`b531234` 及本节新提交需要在你的终端 `git push upstream dribble-clean-v0`。
- latent 架构的接口(盘带 Stage 1′/2′ 在此之后才能开始);阻力接入训练;循环盘带与按周期重锚定;补标 2–4 条 clip;向前 1.0–1.8 m/s 行走数据。


## 15. 循环盘带与重锚定的离线工作(2026-10-16)

### 15.1 循环片段 `scripts/dribble/loop_clip.py`
把一段整数个触球面周期的 clip 重复拼接:片段 [a,b) 从第 i 个触球事件到第 i+L 个,第 k 次重复用刚体 yaw+XY 变换 G^k 摆放(G 把帧 a 的骨盆位姿映到帧 b),所以每个接缝处骨盆位置和朝向连续;四肢只近似周期,所以每次重复开头 10 帧从"原 clip 的真实延续"余弦渐变到重复的起点。球、触球标签、流向锚点帧一并变换/复制;最后一次触球之后的流向字段按数据加载器的约定清空(第一次生成没清,加载器报 `final contact must not have an outgoing flow segment`,已修)。
- 接缝连续性(整数个周期两帧的关节位置差,单位为各关节标准差,随机两帧平均 ~1.35):Reference 1 最好的是从第 6 个事件起(帧 145→225):关节位置 0.45、关节速度 0.61、骨盆局部速度差 0.32 m/s、朝向只转 0.7°;Reference 2 最好的是第 1 个事件起(帧 52→110):0.51 / 0.48 / 0.26 m/s / 2.7°。约为随机两帧差的三分之一到二分之一,**不是严格无缝**。
- 生成:`motions/dribble-loops/ref1_loop_e6x4_r6.npz`(480 帧 9.6 s,24 次触球)、`ref2_loop_e1x2_r10.npz`(580 帧,20 次触球)。
- **Isaac 测试(冻结 Stage1,球每步放回身前,32 个环境,C2 环境,无学习)**:
  - Reference 1 循环:**30/32 走完全部 480 帧(5 个接缝全过)**,2 个摔倒(其一在第 40 帧,可能与起点有关,未追查);周期脉冲出现在 80/160/240/320/400(31–32 个环境)。
  - Reference 2 循环:32/32 摔倒,中位第 41 帧。**不是接缝问题**:原始未拼接的 Reference 2 也是 32/32 摔倒(中位第 97 帧)。
- **重要更正:Stage1(`model_6999.pt`)只在 Reference 1 上训练**(env.yaml 的 `motion_files` 是 `master-aligned-toe-inside-forward/master_001675_001884`)。它从没见过 Reference 2(速度 1.8 m/s,比 Reference 1 的 1.27 快),所以 Reference 2 目前**不能用于 D1/D2/D3**,必须先把 Reference 2 加进 Stage1 训练。之前文档里"Reference 2 可直接套 L0–L5"只在课程语义层面成立,不代表策略能跟踪它。
- 循环片段起点是片段中间的动作(1.2–1.8 m/s),不像原 clip 从慢速起步,reset 时初始状态更难;Reference 2 循环比原始更早摔倒(41 对 97)与此一致,但没有单独验证这个原因。

### 15.2 按周期重锚定 `scripts/dribble/reanchor_check.py`
目标按"episode 内固定映射 A0"摆放时,机器人的跟踪漂移会累积到目标相对机器人的偏差里;在每个周期边界重新对齐一次(周期内目标不动,**不是每步跟随**),偏差只剩周期内的漂移。用冻结 Stage1 在 Reference 1 原始 clip 上的 16 个真实轨迹(球放回身前,走完整条):
| 第 k 次触球 | 固定映射误差 | 按周期重锚定后 |
|---|---|---|
| 1 / 4 / 8 | 0.05 / 0.07 / 0.28 m | 0.00 / 0.06 / 0.23 m |
| 12 / 16 | 0.37 / 0.54 m | 0.19 / 0.24 m |
- 固定映射下偏差一路累积到 0.54 m(约球直径的 2.5 倍);重锚定后上界约 0.24 m,周期边界处为 0。
- 只重锚定了平移,朝向沿用 A0 的(探针没记录 yaw);连朝向一起重锚定只会更好,但没有验证。
- 这是对离线轨迹的计算,**没有接进任何环境或奖励**;接环境要等主线的 latent 接口,且要保证目标只在周期边界更新、周期内不可变(不是被禁止的实时 root 目标)。
- 这回答了主线要的"第 8、16 次触球偏差":固定映射 0.28 / 0.54 m,重锚定 0.23 / 0.24 m。**注意**:这是 Stage1 加"球被放回身前"的设置,不是学会盘带的策略,所以它说明的是"机器人跟踪参考的漂移",不是盘带本身的偏差。

### 15.3 推送
本地领先远程:`b531234`、`1fe722a`、`df0fed5` 及本节提交。服务器 SSH 公钥认证的是 GitHub 账号 `louiskwoklf`,不是 `RamseyPengTianhu`,所以没有用它推送。已生成 `~/users/syf/footmimic-dribble-clean-v0.bundle` 供你在自己的电脑上取走推送。
