# Cricket Ball Bounce & Length Analyzer

Analyze cricket bowling videos to detect ball tracking, identify the bounce point, and classify the delivery length (e.g., Yorker/Full/Good/Short). The tool can process a single video or batch-process a folder, optionally auto-detecting camera view (side vs front) and writing an annotated output video.

## Overview

This project processes cricket delivery footage to:

- **Track the ball over time**
- **Detect whether/where the ball bounces**
- **Classify the delivery “length”** using view-specific heuristics
- **Export results as an output video** (and optionally show a live preview with debug overlays)

The entry point is a small CLI (`main.py`) that calls into the processing pipeline (`src/processor.py`).

## Features

- **CLI for single-video or batch processing** (`main.py`)
  - `--video` for one file, or `--folder` for multiple videos
  - Outputs an annotated `*_result.mp4` per input video

- **Automatic camera view detection (side vs front)** (`src/utils/camera.py`)
  - `detect_view(...)` uses edge-orientation heuristics
  - `--view auto|side|front`

- **Camera quality detection** (`src/utils/camera.py`)
  - `detect_camera_quality(...)` estimates blur using Laplacian variance

- **Bounce + length analysis (side view)** (`src/analysis/side_view.py`)
  - Bounce via **Y-direction reversal** after a minimum descent
  - Length via **X-position** of bounce against configured zones

- **Bounce + length analysis (front view)** (`src/analysis/front_view.py`)
  - Bounce via **sudden radius increase** (ball appears larger as it approaches)
  - Length via **Y-position** at (estimated) bounce

- **Delivery result model** (`src/models/delivery_result.py`)
  - Captures bounce status, length, frame bounds, bounce point, and the track

- **Ball track point model** (`src/models/track_point.py`)
  - Frame-indexed `(x, y, radius)` detections

- **Detection and delivery segmentation scaffolding**
  - Ball detector interface (`src/detection/ball_detector.py`)
  - Delivery segmentation utilities (`src/utils/segmentation.py`)

## Tech stack

- **Language:** Python
- **Core dependencies:**
  - `opencv-python` 
  - `numpy` 

## Getting started

### Prerequisites

- Python environment capable of installing the dependencies listed in `requirements.txt`
- Input videos in a format supported by OpenCV (the CLI searches for `*.mp4`, `*.avi`, `*.mov` when using `--folder`)

### Run

The CLI entry point is `main.py`.

Process a single video:

```bash
python main.py --video path/to/video.mp4 --output data/output_videos
```

Batch-process a folder (default input folder is `data/input_videos`):

```bash
python main.py --folder data/input_videos --output data/output_videos
```

Force a specific camera view (otherwise use `--view auto`):

```bash
python main.py --video path/to/video.mp4 --view side
```

Enable live preview and debug overlay:

```bash
python main.py --video path/to/video.mp4 --show --debug
```

### CLI options

Defined in `main.py`:

- `--video`: Single input video path
- `--folder`: Folder containing multiple videos (default: `data/input_videos`)
- `--view`: `auto | side | front` (default: `auto`)
- `--output`: Output video folder (default: `data/output_videos`)
- `--show`: Live preview
- `--debug`: Debug overlay

## Configuration

Analysis thresholds and heuristics are read from `src/config.py` via `CFG` (used by `src/analysis/side_view.py` and `src/analysis/front_view.py`).

Key configuration usages include:

- Side view bounce reversal thresholds and length zones (`src/analysis/side_view.py`)
  - `CFG["bounce_reversal_px"]`
  - `CFG["min_descent_frames"]`
  - `CFG["length_zones_side"]`

- Front view bounce “size jump” parameters (`src/analysis/front_view.py`)
  - `CFG["front_bounce_window"]`
  - `CFG["front_bounce_size_jump"]`

## Project structure

- `main.py` — CLI entry point; gathers videos and calls `process_video(...)`.
- `src/processor.py` — Processing pipeline entry point (`process_video`).

Core modules:

- `src/detection/` — Ball detection logic
  - `ball_detector.py` — `BallDetector` interface for per-frame detection.

- `src/utils/` — Supporting utilities
  - `camera.py` — camera quality detection + view classification
  - `segmentation.py` — functions to segment detections into deliveries

- `src/analysis/` — Bounce and length classification
  - `side_view.py` — side-view heuristics
  - `front_view.py` — front-view heuristics
  - `trajectory_utils.py` — smoothing helper used by both analyses

- `src/models/` — Data structures
  - `track_point.py` — per-frame track point
  - `delivery_result.py` — per-delivery result container

- `src/visualization/` — Drawing/overlay utilities
  - `draw.py` — rendering helpers for visualization overlays

## License

See `LICENSE`.