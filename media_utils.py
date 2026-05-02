#!/usr/bin/env python3

from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import eo_ir_benchmark as benchmark


def extract_pil_video_frames(
    video_path: Path,
    fps: float,
    max_pixels: int,
    max_frames_per_video: Optional[int],
) -> List[Dict[str, object]]:
    frames = extract_video_frames(video_path, fps=fps, max_pixels=max_pixels, jpeg_quality=None)
    if max_frames_per_video is not None and max_frames_per_video > 0 and len(frames) > max_frames_per_video:
        frames = uniform_subsample(frames, max_frames_per_video)
    if not frames:
        raise benchmark.BenchmarkError(f"No frames extracted from {video_path}")
    return frames


def extract_jpeg_video_frames(
    video_path: Path,
    fps: float,
    max_pixels: int,
    max_frames_per_video: Optional[int],
    jpeg_quality: int,
) -> List[Dict[str, object]]:
    frames = extract_video_frames(video_path, fps=fps, max_pixels=max_pixels, jpeg_quality=jpeg_quality)
    if max_frames_per_video is not None and max_frames_per_video > 0 and len(frames) > max_frames_per_video:
        frames = uniform_subsample(frames, max_frames_per_video)
    if not frames:
        raise benchmark.BenchmarkError(f"No frames extracted from {video_path}")
    return frames


def extract_video_frames(
    video_path: Path,
    fps: float,
    max_pixels: int,
    jpeg_quality: Optional[int],
) -> List[Dict[str, object]]:
    if fps <= 0:
        raise benchmark.BenchmarkError(f"fps must be positive, got {fps}")
    if max_pixels <= 0:
        raise benchmark.BenchmarkError(f"max_pixels must be positive, got {max_pixels}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise benchmark.BenchmarkError(f"Failed to open video: {video_path}")

    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if source_fps <= 0:
            source_fps = 30.0
        frames = extract_frames_sequential(capture, source_fps, fps, max_pixels, jpeg_quality)
    finally:
        capture.release()
    return frames


def extract_frames_sequential(
    capture: cv2.VideoCapture,
    source_fps: float,
    target_fps: float,
    max_pixels: int,
    jpeg_quality: Optional[int],
) -> List[Dict[str, object]]:
    frames = []
    frame_index = 0
    next_sample_time = 0.0
    sample_interval = 1.0 / target_fps
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        timestamp_sec = frame_index / source_fps
        if timestamp_sec + (0.5 / source_fps) >= next_sample_time:
            frames.append(encode_frame(frame, timestamp_sec=timestamp_sec, max_pixels=max_pixels, jpeg_quality=jpeg_quality))
            next_sample_time += sample_interval
        frame_index += 1
    return frames


def encode_frame(
    frame: object,
    timestamp_sec: float,
    max_pixels: int,
    jpeg_quality: Optional[int],
) -> Dict[str, object]:
    resized = resize_to_pixel_budget(frame, max_pixels=max_pixels)
    height, width = resized.shape[:2]
    if jpeg_quality is None:
        from PIL import Image

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).convert("RGB")
        return {
            "timestamp_sec": float(timestamp_sec),
            "width": int(width),
            "height": int(height),
            "image": image,
        }

    quality = max(1, min(int(jpeg_quality), 100))
    ok, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise benchmark.BenchmarkError("Failed to encode video frame as JPEG")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return {
        "timestamp_sec": float(timestamp_sec),
        "width": int(width),
        "height": int(height),
        "data_url": f"data:image/jpeg;base64,{payload}",
    }


def resize_to_pixel_budget(frame: object, max_pixels: int) -> object:
    height, width = frame.shape[:2]
    pixels = width * height
    if pixels <= max_pixels:
        return frame
    scale = (max_pixels / float(pixels)) ** 0.5
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def uniform_subsample(frames: List[Dict[str, object]], max_count: int) -> List[Dict[str, object]]:
    if max_count <= 0 or len(frames) <= max_count:
        return frames
    if max_count == 1:
        return [frames[0]]
    last = len(frames) - 1
    indexes = sorted({int(round(index * last / (max_count - 1))) for index in range(max_count)})
    return [frames[index] for index in indexes]
