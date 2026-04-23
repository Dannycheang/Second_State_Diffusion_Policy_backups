"""Task-4 batch benchmark utility for state/rgbd policy evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from visual_perturbation import load_perturbation_config, perturbation_metadata


def _to_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    return float(value)


def _to_int(value: Any, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    return int(float(value))


def _resolve_path(base_dir: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _read_last_csv_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None

    last_row = None
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            last_row = row
    return last_row


def _read_last_eval_metric(metrics_csv: Path) -> dict[str, str] | None:
    if not metrics_csv.exists():
        return None

    last_row = None
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("eval_success_rate") not in (None, ""):
                last_row = row
    return last_row


def _load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    with manifest_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    return payload.get("benchmarks", [])


def _evaluate_entry(
    entry: dict[str, Any],
    manifest_dir: Path,
    render_backend: str,
    skip_missing_checkpoints: bool,
) -> dict[str, Any]:
    checkpoint_path = _resolve_path(manifest_dir, entry.get("checkpoint"))
    existing_eval_csv = _resolve_path(manifest_dir, entry.get("existing_eval_csv"))
    metrics_csv = _resolve_path(manifest_dir, entry.get("metrics_csv"))
    perturbation_config = load_perturbation_config(
        str(_resolve_path(manifest_dir, entry.get("perturbation_config")))
        if entry.get("perturbation_config")
        else None
    )
    perturb_meta = perturbation_metadata(perturbation_config)

    row: dict[str, Any] = {
        "name": entry.get("name", entry.get("task", "")),
        "task": entry.get("task", ""),
        "env": entry.get("env", ""),
        "checkpoint": str(checkpoint_path) if checkpoint_path else "",
        "policy_input_type": entry.get("policy_input_type", perturb_meta["policy_input_type"]),
        "episodes": entry.get("episodes", ""),
        "max_steps": entry.get("max_steps", ""),
        "action_horizon": entry.get("action_horizon", ""),
        "inference_steps": entry.get("inference_steps", ""),
        "sample_init": entry.get("sample_init", "random"),
        "metrics_source": "",
        "status": "ok",
        "epoch": "",
        "success_rate": "",
        "mean_reward": "",
        "mean_steps": "",
        "notes": entry.get("notes", ""),
        **perturb_meta,
    }

    use_existing_only = bool(entry.get("use_existing_only", False))
    if checkpoint_path and checkpoint_path.exists() and not use_existing_only:
        from evaluate_checkpoint import load_checkpoint
        from train_dp import DEVICE, evaluate_policy

        ckpt, train_args, policy, scheduler, normalizer = load_checkpoint(str(checkpoint_path), DEVICE)
        env_id = entry.get("env") or train_args.get("env", "PegInsertionSide-v1")
        obs_mode = entry.get("obs_mode") or train_args.get("obs_mode", "state")
        action_horizon = _to_int(entry.get("action_horizon"), _to_int(train_args.get("action_horizon"), 8))
        inference_steps = _to_int(entry.get("inference_steps"), _to_int(train_args.get("inference_steps"), 20))
        episodes = _to_int(entry.get("episodes"), 10)
        max_steps = _to_int(entry.get("max_steps"), _to_int(train_args.get("eval_max_steps")))

        result = evaluate_policy(
            env_id=env_id,
            policy_net=policy,
            scheduler=scheduler,
            normalizer=normalizer,
            obs_horizon=int(train_args.get("obs_horizon", 2)),
            pred_horizon=int(train_args.get("pred_horizon", 16)),
            num_episodes=episodes,
            obs_mode=obs_mode,
            device=DEVICE,
            render_backend=render_backend,
            num_inference_steps=inference_steps,
            max_episode_steps=max_steps,
            action_horizon=action_horizon,
            sample_init=entry.get("sample_init", "random"),
        )
        row.update(
            {
                "env": env_id,
                "episodes": episodes,
                "max_steps": max_steps,
                "action_horizon": action_horizon,
                "inference_steps": inference_steps,
                "epoch": ckpt.get("epoch", ""),
                "success_rate": result["success_rate"],
                "mean_reward": result["mean_reward"],
                "mean_steps": result["mean_steps"],
                "metrics_source": "live_eval",
            }
        )
        return row

    if existing_eval_csv and existing_eval_csv.exists():
        eval_row = _read_last_csv_row(existing_eval_csv)
        if eval_row:
            row.update(
                {
                    "env": eval_row.get("env", row["env"]),
                    "episodes": _to_int(eval_row.get("episodes"), _to_int(row["episodes"])),
                    "max_steps": _to_int(eval_row.get("max_steps"), _to_int(row["max_steps"])),
                    "action_horizon": _to_int(eval_row.get("action_horizon"), _to_int(row["action_horizon"])),
                    "inference_steps": _to_int(eval_row.get("inference_steps"), _to_int(row["inference_steps"])),
                    "epoch": eval_row.get("epoch", ""),
                    "success_rate": _to_float(eval_row.get("success_rate")),
                    "mean_reward": _to_float(eval_row.get("mean_reward")),
                    "mean_steps": _to_float(eval_row.get("mean_steps")),
                    "metrics_source": "existing_eval_csv",
                }
            )
            return row

    if metrics_csv and metrics_csv.exists():
        metrics_row = _read_last_eval_metric(metrics_csv)
        if metrics_row:
            row.update(
                {
                    "epoch": metrics_row.get("epoch", ""),
                    "success_rate": _to_float(metrics_row.get("eval_success_rate")),
                    "mean_reward": _to_float(metrics_row.get("eval_mean_reward")),
                    "mean_steps": _to_float(metrics_row.get("eval_mean_steps")),
                    "metrics_source": "metrics_csv",
                }
            )
            return row

    row["status"] = "missing"
    row["metrics_source"] = "missing"
    if checkpoint_path and not checkpoint_path.exists():
        row["notes"] = (row["notes"] + " | checkpoint missing locally").strip(" |")
    if skip_missing_checkpoints:
        return row
    raise FileNotFoundError(f"Missing data source for benchmark entry: {entry.get('name', entry)}")


def _record_entry_video(
    entry: dict[str, Any],
    row: dict[str, Any],
    manifest_dir: Path,
    render_backend: str,
) -> None:
    if not entry.get("record_video", False):
        return

    checkpoint_path = _resolve_path(manifest_dir, entry.get("checkpoint"))
    video_out = _resolve_path(manifest_dir, entry.get("video_out"))
    if not checkpoint_path or not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for video recording: {entry.get('name', entry)}")
    if not video_out:
        raise ValueError(f"record_video=true requires video_out: {entry.get('name', entry)}")

    from types import SimpleNamespace

    from evaluate_checkpoint import load_checkpoint
    from record_policy_video import rollout_to_video
    from train_dp import DEVICE

    _, train_args, policy, scheduler, normalizer = load_checkpoint(str(checkpoint_path), DEVICE)
    perturbation_config = load_perturbation_config(
        str(_resolve_path(manifest_dir, entry.get("perturbation_config")))
        if entry.get("perturbation_config")
        else None
    )
    args = SimpleNamespace(
        env=row.get("env") or entry.get("env") or train_args.get("env", "PickCube-v1"),
        obs_mode=entry.get("obs_mode") or train_args.get("obs_mode", "state"),
        max_steps=_to_int(row.get("max_steps"), _to_int(entry.get("max_steps"), _to_int(train_args.get("eval_max_steps")))),
        action_horizon=_to_int(row.get("action_horizon"), _to_int(entry.get("action_horizon"), _to_int(train_args.get("action_horizon"), 8))),
        inference_steps=_to_int(row.get("inference_steps"), _to_int(entry.get("inference_steps"), _to_int(train_args.get("inference_steps"), 20))),
        sample_init=entry.get("sample_init", "random"),
        render_backend=render_backend,
        fps=float(entry.get("video_fps", 30.0)),
        codec=entry.get("video_codec", "mp4v"),
    )
    seed = _to_int(entry.get("video_seed"), 0) or 0
    success, steps, reward = rollout_to_video(
        args=args,
        policy=policy,
        scheduler=scheduler,
        normalizer=normalizer,
        train_args=train_args,
        seed=seed,
        out_path=video_out,
        perturbation_config=perturbation_config,
    )
    row["video_path"] = str(video_out)
    row["video_seed"] = seed
    row["video_success"] = success
    row["video_steps"] = steps
    row["video_reward"] = reward


def _append_video_index(path: Path, rows: list[dict[str, Any]]) -> None:
    video_rows = []
    for row in rows:
        video_path = row.get("video_path")
        if not video_path:
            continue
        video_rows.append(
            {
                "name": row.get("name", ""),
                "env": row.get("env", ""),
                "policy_input_type": row.get("policy_input_type", ""),
                "base_video": row.get("checkpoint", ""),
                "output_video": video_path,
                "perturbation_name": row.get("perturbation_name", ""),
                "perturbation_type": row.get("perturbation_type", ""),
                "perturbation_level": row.get("perturbation_level", ""),
                "success_rate": row.get("success_rate", ""),
                "notes": row.get("notes", ""),
            }
        )
    if not video_rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(video_rows[0].keys())
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(video_rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Benchmark Summary",
        "",
        "| name | task | policy_input_type | perturbation_type | episodes | success_rate | mean_reward | mean_steps | source | notes |",
        "|---|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {task} | {policy_input_type} | {perturbation_type} | {episodes} | {success_rate} | {mean_reward} | {mean_steps} | {metrics_source} | {notes} |".format(
                **row
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(path: Path, rows: list[dict[str, Any]], include_plot: bool) -> None:
    lines = [
        "# 任务四评估报告",
        "",
        "## 结论",
        "",
        "- 当前仓库已补齐任务四最低交付所需的批量 benchmark 与视觉扰动配置管线。",
        "- 当前基线均为 `state` policy，视觉扰动只影响渲染与视频，不影响策略决策输入，不能据此宣称视觉鲁棒性。",
        "- PickCube 适合作为稳定主 benchmark；Peg 结果只能视为预实验，因为 baseline 尚未稳定收敛。",
        "",
        "## 当前基线汇总",
        "",
        "| name | 任务 | policy_input_type | perturbation_type | episodes | success_rate | mean_reward | mean_steps | 结果来源 | 备注 |",
        "|---|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['task']} | {row['policy_input_type']} | {row['perturbation_type']} | {row['episodes']} | {row['success_rate']} | {row['mean_reward']} | {row['mean_steps']} | {row['metrics_source']} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## 任务四交付物",
            "",
            "- `benchmark_results/benchmark_summary.csv`：统一 benchmark 汇总，包含 `policy_input_type`、扰动类型和配置字段。",
            "- `benchmark_results/benchmark_summary.md`：便于快速查看的 Markdown 摘要。",
            "- `output_visual_perturbation/configs/*.json`：视觉扰动配置记录。",
            "- `output_visual_perturbation/video_index.csv`：扰动前后视频索引。",
            "",
            "## 说明",
            "",
            "- `policy_input_type` 必须在后续所有任务四结果中明确标注为 `state` 或 `rgbd`。",
            "- 如果后续加入真正的视觉策略，请复用本脚本的统一 CSV 输出格式，并把同名字段继续保留。",
            "- 若继续录制扰动视频，请配合 `record_policy_video.py --perturbation-config ... --index-csv ... --save-all-attempts` 输出成功/失败视频与索引。",
            "",
        ]
    )
    if include_plot:
        lines.insert(23, "- `benchmark_results/training_curves_compare.png`：训练曲线对比图。")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_training_curves(output_path: Path, rows: list[dict[str, Any]], manifest: list[dict[str, Any]], manifest_dir: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[plot] skip plotting training curves: {exc}")
        return False

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plotted = 0
    manifest_by_name = {entry.get("name", entry.get("task", "")): entry for entry in manifest}
    for row in rows:
        entry = manifest_by_name.get(row["name"], {})
        metrics_csv = _resolve_path(manifest_dir, entry.get("metrics_csv"))
        if not metrics_csv or not metrics_csv.exists():
            continue

        epochs = []
        train_loss = []
        eval_epochs = []
        eval_success = []
        with metrics_csv.open("r", encoding="utf-8", newline="") as f:
            for metric_row in csv.DictReader(f):
                epochs.append(_to_int(metric_row.get("epoch"), 0))
                train_loss.append(_to_float(metric_row.get("train_loss"), 0.0))
                if metric_row.get("eval_success_rate") not in (None, ""):
                    eval_epochs.append(_to_int(metric_row.get("epoch"), 0))
                    eval_success.append(_to_float(metric_row.get("eval_success_rate"), 0.0))

        if not epochs:
            continue
        label = row["name"]
        axes[0].plot(epochs, train_loss, label=label)
        if eval_epochs:
            axes[1].plot(eval_epochs, eval_success, label=label)
        plotted += 1

    if not plotted:
        plt.close(fig)
        return False

    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Eval Success Rate")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch benchmark policies for task 4.")
    parser.add_argument("--manifest", default="task4_benchmark_manifest.json")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--render-backend", default="sapien_cpu")
    parser.add_argument("--skip-missing-checkpoints", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--video-index-csv", default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest_dir = manifest_path.parent
    manifest = _load_manifest(manifest_path)

    rows = []
    for entry in manifest:
        row = _evaluate_entry(entry, manifest_dir, args.render_backend, args.skip_missing_checkpoints)
        _record_entry_video(entry, row, manifest_dir, args.render_backend)
        rows.append(row)

    output_dir = Path(args.output_dir).resolve()
    _write_csv(output_dir / "benchmark_summary.csv", rows)
    _write_markdown(output_dir / "benchmark_summary.md", rows)
    plotted = False
    if not args.no_plot:
        plotted = _plot_training_curves(output_dir / "training_curves_compare.png", rows, manifest, manifest_dir)
    _write_report(output_dir / "task4_evaluation_report.md", rows, include_plot=plotted)
    if args.video_index_csv:
        _append_video_index(Path(args.video_index_csv).resolve(), rows)

    print(f"wrote benchmark outputs to {output_dir}")


if __name__ == "__main__":
    main()
