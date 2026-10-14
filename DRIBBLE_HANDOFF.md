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
| Ref 2 训练 npz(含球轨迹) | `motions/master-annotated-003178-nodes/master_003178_003326_unitree_g1.npz` |
| Ref 2 无节点版 | `motions/master-annotated-003178/` |
| Ref 2 clean nodes | `motions/contact-nodes-clean-v0-003178/master_003178_003326_unitree_g1.nodes.json`(另有 `.phase_audit.json`、`summary.json`) |
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
- 在此之前,**D1/D2 只应使用 Reference 1**。

### 下一步:D1
- 用 `shell/clean_v0_dribbling_train.sh stage2` 以 `model_6999.pt` 起步,先用 `Tracking-Flat-G1-CleanV0-Stage2-C2-RNN`(设 `S2_TASK`)小规模探针,两条 reference 各一次
- 记录:run 名、checkpoint、有序 2 接触成功率、正确面率、摔倒率、第 8/16 次触球偏差(C2 仅 2 接触,第 8/16 次偏差需在 D2 才有)
- 风险提示:Stage2 的 warm-start 路径(`checkpoint_loading.py`、critic 292→含 S2 上下文的补零)从未在 clean-v0 上实跑
