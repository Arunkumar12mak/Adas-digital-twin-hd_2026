# ADAS Digital Twin HD

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-111F68)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-research%20prototype-orange)](#important-engineering-note)
A Python-based computer-vision pipeline that converts monocular road video into an HD digital-twin visualization for ADAS scenario analysis.

## Features

- YOLO11 object detection with YOLOv8 fallback
- ByteTrack multi-object tracking
- Bird's-eye-view lane reconstruction and persistent lane-line tracking
- Ego and actor lane-change, cut-in, and cut-out detection
- Lead-vehicle selection, relative-speed estimation, and TTC analysis
- Traffic-light, crosswalk, stop-line, and junction recognition
- HD visualization with CSV and JSON engineering-data exports

## Repository Structure

```text
adas-digital-twin-hd/
├── src/
│   └── adas_digital_twin_hd.py
├── sample_data/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
├── .gitignore
├── CONTRIBUTING.md
├── README.md
└── requirements.txt
```

## Requirements

- Python 3.10 or later recommended
- A road-scene video in a format supported by OpenCV
- GPU is optional, but recommended for faster inference

## Installation

```bash
git clone <your-repository-url>
cd adas-digital-twin-hd
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# Linux or macOS
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

```bash
python src/adas_digital_twin_hd.py
```

The application prompts you to select a road video, choose an output folder, and select automatic, manual, or default lane calibration.

Preview controls:

- `p`: pause or resume
- `q`: stop processing

## Main Outputs

Depending on the selected output folder, the application can generate:

- `adas_digital_twin_hd.mp4`
- `tracks.csv`
- `lanes.csv`
- `events.csv`
- `events.json`
- `summary.json`

## Configuration

The main configuration values are located near the top of `src/adas_digital_twin_hd.py`. Review the model weights, detection thresholds, lane width, camera field of view, camera height, calibration settings, theme, preview mode, and output resolution before running a new dataset.

## Important Engineering Note

Monocular video does not provide an absolute metric scale without accurate camera calibration. Distance, speed, lane geometry, and TTC values are engineering estimates. This project is intended for development, visualization, and scenario analysis, not production safety validation or regulatory pass/fail assessment.

## Model Files and Sample Videos

YOLO weight files, road videos, and generated outputs are excluded from Git because they can be large. Add a small, non-confidential sample clip to `sample_data/` only if you have permission to publish it. Do not upload proprietary test data, vehicle-registration details, faces, or customer information.

## License

This project is released under the [MIT License](LICENSE). Review organizational and intellectual-property requirements before publishing company-related code or data.
