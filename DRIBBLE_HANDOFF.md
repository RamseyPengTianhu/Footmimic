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

Reference 2 已完成:人工标注 → 球节点合成 → clean node 编译 → replay 视频。

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

## 7. 待确认 / 风险

1. **Ref 2 节点相位审计**:`summary.json` 显示 `nearest_intent_mismatch_rate = 0.306`(75/245 帧),
   即非接触帧的最近邻特征常落在不同事件意图上。是否影响 Stage2 需要判断。
2. **Curriculum 与 Ref 2 的模式不匹配**:C4/T4 写死 O/O/I/I 单元,而 Ref 2 是严格 I/O 交替。
   Ref 2 不能直接套用 C4/T4,需要泛化 curriculum 或只用 C2/T2/C3 级别。
3. 所有代码和数据改动(约 7400 行新增)与新 motions 均未提交,建议先 commit 再让别的对话接手。
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

## 9. 下一步建议

1. 判断 phase audit 30.6% 的影响;必要时修正节点或标注
2. 决定 Ref 2 的 curriculum 适配方式(见风险 2)
3. 以 `model_6999.pt` 为起点启动 Stage2(先 C2/T2 小规模探针),评估触球成功率
4. 提交当前改动,并把旧 README 顶部加一行指向本文件
