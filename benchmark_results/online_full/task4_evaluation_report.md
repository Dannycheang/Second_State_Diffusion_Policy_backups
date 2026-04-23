# 任务四正式评估报告

## 一、任务目标

任务四的目标是补齐统一评估、视觉扰动配置、视频记录与结果汇总流程，并基于当前项目已有 checkpoint 做在线评估。

根据项目交接文档，当前任务二交付的策略均为 `state` policy，即策略输入是状态向量而不是图像。因此本阶段任务四的结果应理解为：

1. 已完成视觉扰动评估管线与在线复评流程。
2. 已验证扰动配置、环境运行、视频录制和统一结果汇总可以正常工作。
3. 不能据此直接声称“视觉输入策略具有鲁棒性”。

## 二、评估边界与结论口径

本报告必须遵守以下结论边界：

1. 当前所有在线评估对象的 `policy_input_type` 均为 `state`。
2. 光照、相机偏移、背景色等扰动只改变渲染画面与录制视频，不直接进入当前策略决策输入。
3. 因此，本报告验证的是“任务四评估管线与扰动流程可运行”，而不是“视觉策略鲁棒性已被证明”。
4. `PegInsertionSide-v1` 当前仍未稳定收敛，因此其结果只能作为预实验，不得写成正式视觉鲁棒性结论。

## 三、评估设置

### 3.1 PickCube 正式在线评估

- checkpoint：`output_pickcube_state_final/checkpoints/policy_best.pt`
- policy_input_type：`state`
- baseline 配置：`episodes=10`、`max_steps=100`、`action_horizon=8`、`inference_steps=20`
- 扰动配置：
  - `output_visual_perturbation/configs/pickcube_lighting_low.json`
  - `output_visual_perturbation/configs/pickcube_camera_shift_01.json`
  - `output_visual_perturbation/configs/pickcube_background_color_01.json`

### 3.2 Peg 预实验在线评估

- checkpoint：`output_peg_state_motionplanning_resume/checkpoints/policy_best.pt`
- policy_input_type：`state`
- baseline 配置：`episodes=30`、`max_steps=350`、`action_horizon=10`、`inference_steps=20`
- 扰动配置：
  - `output_visual_perturbation/configs/peg_lighting_low_preview.json`

## 四、结果汇总

| 条目 | 任务 | 输入类型 | 扰动 | episodes | success_rate | mean_reward | mean_steps | 结论口径 |
|---|---|---|---|---:|---:|---:|---:|---|
| `pickcube_online_baseline` | PickCube | state | 无 | 10 | 70.0% | 21.06 | 72.0 | 可作为任务四主 benchmark |
| `pickcube_online_lighting_low` | PickCube | state | lighting | 10 | 80.0% | 19.21 | 67.4 | 仅说明扰动流程与在线评估可运行 |
| `pickcube_online_camera_shift_01` | PickCube | state | camera | 10 | 80.0% | 19.21 | 67.4 | 仅说明扰动流程与在线评估可运行 |
| `pickcube_online_background_color_01` | PickCube | state | background | 10 | 80.0% | 19.21 | 67.4 | 仅说明扰动流程与在线评估可运行 |
| `peg_online_baseline_preview` | PegInsertionSide | state | 无 | 30 | 30.0% | 106.01 | 291.53 | 仅为预实验，baseline 未稳定收敛 |
| `peg_online_lighting_low_preview` | PegInsertionSide | state | lighting | 30 | 43.33% | 98.41 | 265.37 | 仅为预实验，不能写成视觉鲁棒性结论 |

## 五、结果解读

### 5.1 PickCube

`PickCube-v1` 目前仍然是任务四最适合使用的主评估对象，因为它是当前项目里唯一已经较稳定跑通的 baseline。  
在本次正式在线评估中，baseline 与三个扰动条目都能正常完成 rollout，并成功生成对应视频，说明：

- 任务四在线评估脚本已经能直接调用真实 checkpoint 做 live eval。
- 扰动配置已经能稳定作用到渲染与视频输出。
- 统一 CSV / Markdown 报告 / 视频索引的交付链路已经打通。

但由于策略输入仍是 `state`，所以 PickCube 的 lighting / camera / background 结果只能说明“渲染端扰动没有破坏当前 state baseline 的整体运行流程”，不能说明图像输入策略对这些视觉变化具有鲁棒性。

### 5.2 Peg

`PegInsertionSide-v1` 的 30 episode baseline 结果仍应维持项目文档中的正式口径：当前模型是可运行 baseline，而不是稳定收敛模型。  
这次加入 lighting preview 后虽然得到了一个数值结果，但该结果仍然不能作为正式视觉鲁棒性证据，原因有两点：

1. 当前 Peg 策略仍然是 `state` policy。
2. Peg baseline 本身尚未稳定收敛，结果波动较大。

因此，Peg 在任务四中应继续保留为“预实验”或“流程验证”对象，而不是主结论对象。

## 六、已生成交付物

### 6.1 正式结果文件

- `benchmark_results/online_full/benchmark_summary.csv`
- `benchmark_results/online_full/benchmark_summary.md`
- `benchmark_results/online_full/task4_evaluation_report.md`

### 6.2 扰动配置

- `output_visual_perturbation/configs/pickcube_lighting_low.json`
- `output_visual_perturbation/configs/pickcube_camera_shift_01.json`
- `output_visual_perturbation/configs/pickcube_background_color_01.json`
- `output_visual_perturbation/configs/peg_lighting_low_preview.json`

### 6.3 在线生成视频

- `output_pickcube_analysis/videos/success_seed_0_pickcube_none.mp4`
- `output_pickcube_analysis/videos/success_seed_0_pickcube_pickcube_lighting_low.mp4`
- `output_pickcube_analysis/videos/success_seed_0_pickcube_pickcube_camera_shift_01.mp4`
- `output_pickcube_analysis/videos/success_seed_0_pickcube_pickcube_background_color_01.mp4`
- `output_peg_analysis/videos/success_seed_9_peginsertionside_none.mp4`
- `output_peg_analysis/videos/success_seed_9_peginsertionside_peg_lighting_low_preview.mp4`

### 6.4 视频索引

- `output_visual_perturbation/video_index.csv`

## 七、最终结论

本阶段任务四已经完成了“统一在线评估 + 扰动配置 + 视频记录 + 结果汇总”的正式交付要求。  
当前最准确的结论应写为：

- 任务四在线评估管线已经搭建完成，并可在本机 `.venv` 环境下直接运行。
- PickCube 可作为当前任务四的主 benchmark。
- Peg 只能作为预实验对象，不应写成稳定收敛或视觉鲁棒性已验证。
- 若要得到真正的视觉扰动鲁棒性结论，下一步必须训练或接入 `rgbd` / image-based policy。
