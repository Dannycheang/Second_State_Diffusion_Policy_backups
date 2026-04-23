# Benchmark Summary

| name | task | policy_input_type | perturbation_type | episodes | success_rate | mean_reward | mean_steps | source | notes |
|---|---|---|---|---:|---:|---:|---:|---|---|
| pickcube_online_baseline | PickCube | state | none | 10 | 70.0 | 23.379395438358188 | 72.8 | live_eval | Formal online baseline evaluation for task 4. |
| pickcube_online_lighting_low | PickCube | state | lighting | 10 | 80.0 | 19.211186687648297 | 67.4 | live_eval | Formal online lighting preview for task 4. State policy input is unchanged. |
| pickcube_online_camera_shift_01 | PickCube | state | camera | 10 | 80.0 | 19.211186687648297 | 67.4 | live_eval | Formal online camera preview for task 4. State policy input is unchanged. |
| pickcube_online_background_color_01 | PickCube | state | background | 10 | 80.0 | 19.211186687648297 | 67.4 | live_eval | Formal online background-color preview for task 4. State policy input is unchanged. |
| peg_online_baseline_preview | PegInsertionSide | state | none | 30 | 30.0 | 106.00914299550156 | 291.53333333333336 | live_eval | Peg preview only. Baseline is not stably converged. |
| peg_online_lighting_low_preview | PegInsertionSide | state | lighting | 30 | 43.333333333333336 | 98.40898178035859 | 265.3666666666667 | live_eval | Peg preview only. Render-only perturbation does not demonstrate visual robustness. |
