import csv
from pathlib import Path

import matplotlib
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs  # noqa: F401

from evaluate_checkpoint import load_checkpoint
from train_dp import DEVICE

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def to_bool(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


def as_int(raw: str, default: int) -> int:
    if raw in (None, ""):
        return default
    return int(float(raw))


def extract_state_obs(raw_obs) -> np.ndarray:
    state_obs = raw_obs["state"] if isinstance(raw_obs, dict) else raw_obs
    state_obs = np.asarray(state_obs, dtype=np.float32)
    return state_obs.reshape(-1)


def rollout_action_trace(row: dict, root: Path) -> dict:
    checkpoint = Path(row["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    _, train_args, policy, scheduler, normalizer = load_checkpoint(str(checkpoint), DEVICE)
    env_id = row["env"]
    obs_mode = "state"
    obs_horizon = int(train_args.get("obs_horizon", 2))
    pred_horizon = int(train_args.get("pred_horizon", 16))
    max_steps = as_int(row.get("max_steps", ""), as_int(str(train_args.get("eval_max_steps", 500)), 500))
    action_horizon = as_int(row.get("action_horizon", ""), as_int(str(train_args.get("action_horizon", 8)), 8))
    inference_steps = as_int(row.get("inference_steps", ""), as_int(str(train_args.get("inference_steps", 20)), 20))
    seed = as_int(row.get("video_seed", ""), 0)
    sample_init = row.get("sample_init", "random") or "random"

    env = gym.make(
        env_id,
        obs_mode=obs_mode,
        render_mode="rgb_array",
        render_backend="sapien_cpu",
        max_episode_steps=max_steps,
    )
    obs, _ = env.reset(seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    first_obs = extract_state_obs(obs)
    obs_history = [first_obs.copy() for _ in range(obs_horizon)]
    actions = []
    done = False
    step = 0
    total_reward = 0.0
    success = False

    policy.eval()
    with torch.no_grad():
        while not done and step < max_steps:
            obs_seq = np.stack(obs_history[-obs_horizon:], axis=0)
            obs_t = torch.from_numpy(obs_seq).float().unsqueeze(0).to(DEVICE)
            obs_t = normalizer.normalize_obs(obs_t)

            action_seq = scheduler.sample(
                policy,
                obs_t,
                pred_horizon,
                normalizer.action_dim,
                num_inference_steps=inference_steps,
                init_noise=sample_init,
            )
            action_seq = normalizer.denormalize_action(action_seq)

            exec_horizon = min(action_horizon, pred_horizon)
            for action_i in range(exec_horizon):
                action_np = action_seq[0, action_i].cpu().numpy()
                action_np = np.clip(action_np, -1, 1).astype(np.float32)
                actions.append(action_np.copy())
                obs, reward, terminated, truncated, info = env.step(action_np)
                total_reward += float(reward)
                step += 1
                success = bool(info.get("success", False))
                done = bool(terminated or truncated or success)
                obs_history.append(extract_state_obs(obs))
                obs_history.pop(0)
                if done or step >= max_steps:
                    break

    env.close()
    if not actions:
        return {
            "actions": np.zeros((0, normalizer.action_dim), dtype=np.float32),
            "success": success,
            "steps": step,
            "reward": total_reward,
        }
    return {
        "actions": np.stack(actions, axis=0),
        "success": success,
        "steps": step,
        "reward": total_reward,
    }


def write_task_stats(task_key: str, rows: list[dict], root: Path) -> None:
    out_dir = root / ("output_pickcube_analysis" if task_key == "pickcube" else "output_peg_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "action_stats.csv"
    out_png = out_dir / "action_stats.png"

    dim_rows = []
    summary_rows = []
    for row in rows:
        trace = rollout_action_trace(row, root)
        actions = trace["actions"]
        if actions.shape[0] == 0:
            continue
        tail = actions[-50:] if actions.shape[0] >= 50 else actions
        name = row["name"]
        success = trace["success"]
        steps = trace["steps"]
        reward = trace["reward"]

        summary_rows.append(
            {
                "name": name,
                "steps": steps,
                "reward": reward,
                "success": success,
                "mean_abs_action": float(np.mean(np.abs(actions))),
                "max_abs_action": float(np.max(np.abs(actions))),
                "sat_ratio_abs_ge_0.99": float(np.mean(np.abs(actions) >= 0.99)),
                "tail50_mean_abs_action": float(np.mean(np.abs(tail))),
                "tail50_max_abs_action": float(np.max(np.abs(tail))),
            }
        )

        for dim in range(actions.shape[1]):
            a = actions[:, dim]
            t = tail[:, dim]
            dim_rows.append(
                {
                    "name": name,
                    "action_dim": dim,
                    "count": int(a.shape[0]),
                    "mean": float(np.mean(a)),
                    "std": float(np.std(a)),
                    "max_abs": float(np.max(np.abs(a))),
                    "sat_ratio_abs_ge_0.99": float(np.mean(np.abs(a) >= 0.99)),
                    "tail50_mean_abs": float(np.mean(np.abs(t))),
                    "tail50_max_abs": float(np.max(np.abs(t))),
                    "success": success,
                    "steps": steps,
                    "reward": reward,
                }
            )

    fieldnames = [
        "name",
        "action_dim",
        "count",
        "mean",
        "std",
        "max_abs",
        "sat_ratio_abs_ge_0.99",
        "tail50_mean_abs",
        "tail50_max_abs",
        "success",
        "steps",
        "reward",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dim_rows)

    if summary_rows:
        labels = [r["name"] for r in summary_rows]
        x = np.arange(len(labels))
        mean_abs = [r["mean_abs_action"] for r in summary_rows]
        sat_ratio = [r["sat_ratio_abs_ge_0.99"] for r in summary_rows]
        tail_abs = [r["tail50_mean_abs_action"] for r in summary_rows]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].bar(x - 0.15, mean_abs, width=0.3, label="mean_abs")
        axes[0].bar(x + 0.15, tail_abs, width=0.3, label="tail50_mean_abs")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=20, ha="right")
        axes[0].set_title(f"{task_key} action magnitude")
        axes[0].grid(True, axis="y", alpha=0.3)
        axes[0].legend()

        axes[1].bar(x, sat_ratio, width=0.4, color="tab:orange")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=20, ha="right")
        axes[1].set_ylim(0, 1)
        axes[1].set_title(f"{task_key} saturation ratio |a|>=0.99")
        axes[1].grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

    # extra summary csv to help quick checks
    summary_csv = out_dir / "action_stats_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "steps",
                "reward",
                "success",
                "mean_abs_action",
                "max_abs_action",
                "sat_ratio_abs_ge_0.99",
                "tail50_mean_abs_action",
                "tail50_max_abs_action",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[{task_key}] wrote {out_csv}")
    print(f"[{task_key}] wrote {out_png}")
    print(f"[{task_key}] wrote {summary_csv}")


def main() -> None:
    root = Path(__file__).resolve().parent
    benchmark_csv = root / "benchmark_results" / "online_full" / "benchmark_summary.csv"
    if not benchmark_csv.exists():
        raise FileNotFoundError(f"missing benchmark csv: {benchmark_csv}")

    pick_rows = []
    peg_rows = []
    with benchmark_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            task = str(row.get("task", "")).strip().lower()
            if task == "pickcube":
                pick_rows.append(row)
            elif task == "peginsertionside":
                peg_rows.append(row)

    write_task_stats("pickcube", pick_rows, root)
    write_task_stats("peginsertionside", peg_rows, root)


if __name__ == "__main__":
    main()
