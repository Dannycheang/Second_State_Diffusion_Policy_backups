import csv
from pathlib import Path


def to_bool(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


def norm_task(task: str) -> str:
    t = str(task).strip().lower()
    if t == "pickcube":
        return "pickcube"
    if t == "peginsertionside":
        return "peginsertionside"
    return t


def build_new_video_path(root: Path, row: dict) -> Path:
    task = norm_task(row.get("task", ""))
    seed = str(row.get("video_seed", "na")).strip() or "na"
    success = to_bool(row.get("video_success", "false"))
    perturb = str(row.get("perturbation_name", "none")).strip().lower() or "none"
    label = "success" if success else "fail"

    if task == "pickcube":
        base_dir = root / "output_pickcube_analysis" / "videos"
    elif task == "peginsertionside":
        base_dir = root / "output_peg_analysis" / "videos"
    else:
        return Path(row.get("video_path", ""))

    filename = f"{label}_seed_{seed}_{task}_{perturb}.mp4"
    return (base_dir / filename).resolve()


def update_benchmark_summary(root: Path) -> list[dict]:
    summary_csv = root / "benchmark_results" / "online_full" / "benchmark_summary.csv"
    with summary_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []

    for row in rows:
        new_path = build_new_video_path(root, row)
        if new_path.exists():
            row["video_path"] = str(new_path)

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_visual_video_index(root: Path, rows: list[dict]) -> Path:
    out_csv = root / "output_visual_perturbation" / "video_index.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "env",
        "policy_input_type",
        "base_video",
        "output_video",
        "perturbation_name",
        "perturbation_type",
        "perturbation_level",
        "success_rate",
        "notes",
    ]
    out_rows = []
    for row in rows:
        out_rows.append(
            {
                "name": row.get("name", ""),
                "env": row.get("env", ""),
                "policy_input_type": row.get("policy_input_type", ""),
                "base_video": row.get("checkpoint", ""),
                "output_video": row.get("video_path", ""),
                "perturbation_name": row.get("perturbation_name", ""),
                "perturbation_type": row.get("perturbation_type", ""),
                "perturbation_level": row.get("perturbation_level", ""),
                "success_rate": row.get("success_rate", ""),
                "notes": row.get("notes", ""),
            }
        )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    return out_csv


def update_task4_report(root: Path, rows: list[dict]) -> Path:
    report = root / "benchmark_results" / "online_full" / "task4_evaluation_report.md"
    text = report.read_text(encoding="utf-8")

    marker_start = "### 6.3 在线生成视频\n"
    marker_end = "### 6.4 视频索引\n"
    start_idx = text.find(marker_start)
    end_idx = text.find(marker_end)
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return report

    lines = []
    ordered = [
        "pickcube_online_baseline",
        "pickcube_online_lighting_low",
        "pickcube_online_camera_shift_01",
        "peg_online_baseline_preview",
        "peg_online_lighting_low_preview",
    ]
    by_name = {r.get("name", ""): r for r in rows}
    for name in ordered:
        row = by_name.get(name)
        if not row:
            continue
        lines.append(f"- `{row.get('video_path', '')}`")

    replacement = marker_start + "\n" + "\n".join(lines) + "\n\n"
    new_text = text[:start_idx] + replacement + text[end_idx:]
    report.write_text(new_text, encoding="utf-8")
    return report


def main() -> None:
    root = Path(__file__).resolve().parent
    rows = update_benchmark_summary(root)
    out_index = write_visual_video_index(root, rows)
    report = update_task4_report(root, rows)
    print(f"updated benchmark summary: {root / 'benchmark_results' / 'online_full' / 'benchmark_summary.csv'}")
    print(f"wrote visual index: {out_index}")
    print(f"updated report: {report}")


if __name__ == "__main__":
    main()
