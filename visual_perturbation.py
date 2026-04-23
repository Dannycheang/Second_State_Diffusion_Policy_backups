"""Utilities for task-4 visual perturbation configs and render-only effects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _as_serialized_text(value: Any) -> str:
    if value in (None, "", {}, []):
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def load_perturbation_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {
            "enabled": False,
            "name": "none",
            "policy_input_type": "state",
            "perturbation_type": "none",
            "perturbation_level": "none",
            "frame_effects": {},
        }

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    config.setdefault("enabled", True)
    config.setdefault("name", path.stem)
    config.setdefault("policy_input_type", "state")
    config.setdefault("perturbation_type", "custom")
    config.setdefault("perturbation_level", "custom")
    config.setdefault("frame_effects", {})
    return config


def perturbation_metadata(config: dict[str, Any]) -> dict[str, str]:
    return {
        "policy_input_type": str(config.get("policy_input_type", "state")),
        "perturbation_enabled": str(bool(config.get("enabled", False))).lower(),
        "perturbation_name": str(config.get("name", "none")),
        "perturbation_type": str(config.get("perturbation_type", "none")),
        "perturbation_level": str(config.get("perturbation_level", "none")),
        "camera_config": _as_serialized_text(config.get("camera_config")),
        "lighting_config": _as_serialized_text(config.get("lighting_config")),
        "texture_background_config": _as_serialized_text(
            config.get("texture_background_config") or config.get("background_config")
        ),
        "perturbation_notes": str(config.get("notes", "")),
    }


def _mix_with_gray(frame: np.ndarray, saturation: float) -> np.ndarray:
    gray = frame.mean(axis=2, keepdims=True)
    return gray + saturation * (frame - gray)


def _apply_translation(frame: np.ndarray, tx: float, ty: float, border_color: tuple[int, int, int]) -> np.ndarray:
    h, w = frame.shape[:2]
    matrix = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
    return cv2.warpAffine(
        frame,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_color,
    )


def _apply_zoom(frame: np.ndarray, zoom: float, border_color: tuple[int, int, int]) -> np.ndarray:
    if zoom == 1.0:
        return frame

    h, w = frame.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, zoom)
    return cv2.warpAffine(
        frame,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_color,
    )


def apply_frame_perturbation(
    frame: np.ndarray,
    config: dict[str, Any] | None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if not config or not config.get("enabled", False):
        return frame

    frame = np.asarray(frame, dtype=np.float32)
    effects = config.get("frame_effects", {})
    rng = rng or np.random.default_rng()

    border_color = tuple(int(x) for x in effects.get("border_color_rgb", [0, 0, 0]))

    brightness = float(effects.get("brightness", 0.0))
    contrast = float(effects.get("contrast", 1.0))
    frame = frame * contrast + brightness

    saturation = float(effects.get("saturation", 1.0))
    if saturation != 1.0:
        frame = _mix_with_gray(frame, saturation)

    gamma = float(effects.get("gamma", 1.0))
    if gamma > 0 and gamma != 1.0:
        normalized = np.clip(frame / 255.0, 0.0, 1.0)
        frame = np.power(normalized, gamma) * 255.0

    tint_rgb = effects.get("tint_rgb")
    if tint_rgb is not None:
        frame = frame + np.array(tint_rgb, dtype=np.float32).reshape(1, 1, 3)

    overlay_rgb = effects.get("overlay_color_rgb")
    overlay_alpha = float(effects.get("overlay_alpha", 0.0))
    if overlay_rgb is not None and overlay_alpha > 0:
        overlay = np.array(overlay_rgb, dtype=np.float32).reshape(1, 1, 3)
        frame = (1.0 - overlay_alpha) * frame + overlay_alpha * overlay

    noise_std = float(effects.get("gaussian_noise_std", 0.0))
    if noise_std > 0:
        frame = frame + rng.normal(0.0, noise_std, size=frame.shape)

    frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)

    tx = float(effects.get("translate_x", 0.0))
    ty = float(effects.get("translate_y", 0.0))
    if tx or ty:
        frame = _apply_translation(frame, tx, ty, border_color)

    zoom = float(effects.get("zoom", 1.0))
    if zoom != 1.0:
        frame = _apply_zoom(frame, zoom, border_color)

    return frame


def apply_perturbation_to_video(
    input_video: str | Path,
    output_video: str | Path,
    config: dict[str, Any] | None,
    seed: int = 0,
) -> None:
    """Apply render-only perturbations to an existing video file."""
    input_path = Path(input_video)
    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open output video: {output_path}")

    rng = np.random.default_rng(seed)
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            perturbed_rgb = apply_frame_perturbation(frame_rgb, config, rng)
            perturbed_bgr = cv2.cvtColor(perturbed_rgb, cv2.COLOR_RGB2BGR)
            writer.write(perturbed_bgr)
    finally:
        capture.release()
        writer.release()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Apply task-4 visual perturbations to an existing video.")
    parser.add_argument("--config", required=True, help="Path to the perturbation JSON config.")
    parser.add_argument("--input-video", required=True, help="Input MP4 video path.")
    parser.add_argument("--output-video", required=True, help="Output MP4 video path.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for noise-based effects.")
    args = parser.parse_args()

    config = load_perturbation_config(args.config)
    apply_perturbation_to_video(args.input_video, args.output_video, config=config, seed=args.seed)
    print(f"saved perturbed video to {Path(args.output_video).resolve()}")


if __name__ == "__main__":
    main()
