# Vision-Based Attendance System

An automated classroom attendance tracking system using computer vision, face recognition, and pose estimation. The system tracks students in real-time through video or live camera feed, identifies them using hybrid face + appearance embeddings, monitors hand raises, and generates detailed attendance reports.

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
  - [Step 1: Enroll Students](#step-1-enroll-students)
  - [Step 2: Run Attendance Tracking](#step-2-run-attendance-tracking)
- [Command-Line Arguments](#command-line-arguments)
- [Attendance Tracking Modes](#attendance-tracking-modes)
- [Output Files](#output-files)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)

---

## Features

### Core Capabilities
- **Real-time Face Recognition**: Identifies students using pre-enrolled face embeddings with FaceNet
- **Hybrid Tracking**: Combines face recognition with appearance-based features with InceptionResnetV1 pretrained on vggface2 and OSNet 
- **Hand Raise Detection**: Automatically detects and counts when students raise their hands using YOLO26n-pose
- **Flexible Input Sources**: Works with both video files and live camera feeds
- **Checkpoint-Based Attendance**: Uses multiple time-based checkpoints to verify attendance
- **Detailed Reporting**: Generates JSON and human-readable text reports with attendance metrics

### Advanced Features
- **Multi-modal Recognition**: Concatenates face embeddings (512D) and appearance embeddings (512D) with configurable weights
- **Tracking Persistence**: Maintains student identity across frames even when face is temporarily obscured
- **Hand Raise Cooldown**: Prevents duplicate raised-hand-detection with a configurable cooldown period
- **GPU Acceleration**: Supports CUDA, CPU, and MPS devices
- **Live Visualization**: Optional real-time display with annotated video as it generates

---

## How It Works

### Overall Pipeline

```
Video/Camera Input
      ↓
[YOLO Person Detection] → Track each person in the frame
      ↓
[Face Detection (MTCNN)] → Extract face region (cropping)
      ↓
[Face Embedding (InceptionResnetV1 pretrained on vggface2)] → 512D face vector
      ↓
[Appearance Embedding (OSNet)] → 512D appearance vector
      ↓
[Hybrid Embedding] → Weighted combination (default: 40% face + 60% appearance, adjustable with flags)
      ↓
[Student Recognition] → Match against enrolled database (defaulted to models/known_faces.pkl)
      ↓
[Attendance Checkpoint] → Record presence during automatically generated checkpoint windows (defaults to 5 evenly spread)
      ↓
[Hand Pose Detection] → Detect raised hands using YOLO26n-pose keypoint analysis
      ↓
[Output Generation] → Save annotated video (mp4) + attendance report (human-readable .txt and computer-readable .json)
```

### Recognition Strategy

The system uses a **hybrid embedding approach** to maximize recognition accuracy:

1. **Face Embedding** (InceptionResnetV1 on VGGFace2):
   - Strong for facial recognition
   - Can be affected by pose, lighting, occlusion
   - Weight: 40% (default, configurable)

2. **Appearance Embedding** (OSNet):
   - Captures clothing, body shape, posture
   - More robust when face is turned away
   - Weight: 60% (default, configurable)

3. **Fallback Mechanism**:
   - If face is not detected in current frame, uses last known good face embedding
   - Continues tracking using appearance features
   - Prevents loss of identity during temporary occlusions

### Attendance Verification

**Video Mode (5 Checkpoints)**:
- Divides video into 5 equally-spaced checkpoints
- For each checkpoint, creates a detection window (±15 frames by default)
- Student must be detected within at least 80% (by default) of checkpoint windows to be marked present

**Camera Mode (Interval-Based)**:
- Creates new checkpoint every N seconds (default: 10s)
- Each checkpoint has a detection window (default: 15 frames)
- Student must be detected in 80% (by default) of checkpoints to be marked present

---

## System Architecture

### Core Modules

1. **tracking/tracker.py**: `StudentTracker`
   - Manages unique student identities across frames
   - De-duplicates detections using embedding similarity

2. **detectors/pose_rule_based.py**: `PoseRuleBasedDetector`
   - Uses YOLO pose keypoints to detect hand raises
   - **Rule-based analysis** of shoulder, elbow, wrist positions

3. **recognition/face_recognizer.py**: `FaceRecognizer`
   - Matches face embeddings against enrolled database
   - Uses cosine similarity with configurable threshold

### Models Used

| Model                 | Purpose | Size |
|-----------------------|---------|------|
| **YOLOv26n**          | Person detection | ~6MB |
| **YOLO26n-pose**      | Pose estimation (17 keypoints) | ~6MB |
| **MTCNN**             | Face detection | ~2MB |
| **InceptionResnetV1** | Face embedding | ~110MB |
| **OSNet x0.75**       | Appearance embedding | ~3MB |

---

## Installation

### Requirements

```bash
--extra-index-url https://download.pytorch.org/whl/cu121
facenet_pytorch==2.6.0
numpy<2.0.0
opencv-contrib-python==4.9.0.80
opencv-python==4.9.0.80
# macOS (Apple Silicon/Intel): use default PyPI wheels (CPU/MPS)
torch==2.2.2; platform_system == "Darwin"
torchvision==0.17.2; platform_system == "Darwin"
# Linux: use CUDA 12.1 wheels from PyTorch extra index
torch==2.2.2+cu121; platform_system == "Linux"
torchvision==0.17.2+cu121; platform_system == "Linux"
torchreid==0.2.5
ultralytics==8.4.42
gdown
tensorboard
```

### Download Pre-trained Models

The system will automatically download most models on first run. For the appearance model:

```bash
# Download OSNet model
mkdir -p models
wget https://github.com/KaiyangZhou/deep-person-reid/releases/download/v1.0.0/osnet_x0_75_imagenet.pth \
     -O models/osnet_x0_75_imagenet.pth
```

---

## Directory Structure

```
project/
├── main.py                         # Main attendance tracking script
├── enroll_faces.py                 # Face enrollment script
├── tracking/
│   └── tracker.py                  # StudentTracker class
├── detectors/
│   └── base.py                     # Allows for new methods of hand-raise detection to extend a base class requiring a detect function
│   └── pose_rule_based.py          # Hand raise detection
├── recognition/
│   └── face_recognizer.py          # Face matching
├── models/
│   ├── known_faces.pkl             # Enrolled face database
├── StudentIDs/
│   ├── Robert Figueroa/                   # Student folder (used to determine what students are available for attendance metrics)
│   ├── Trace Capel/                 # Having data inside them is optional and not read by the system
│   └── ...
└── output/
    ├── video_annotated.mp4         # Processed video
    ├── video_annotated.json        # Attendance data (JSON)
    └── video_annotated.txt         # Attendance report (text)
```

---

## Usage

### Step 1: Enroll Students

Before running attendance tracking, you need to enroll student faces into the system.

```bash
python enroll_faces.py --name "Robert_Figueroa"
```

**Interactive Process**:
1. Script opens your webcam
2. Position your face in view (green box appears when detected)
3. Press **SPACE** to capture a sample (need 10 samples)
4. Move head slightly between captures for variety
5. Script automatically exits after 10 samples
6. Face embedding is saved to `models/new_known_faces.pkl`

**Tips for Good Enrollment**:
- Ensure good lighting
- Capture from different angles (±15° rotation)
- Include slight variations in expression
- Avoid extreme poses or occlusions

**Enroll Multiple Students**:
```bash
python enroll_faces.py --name "Robert_Figueroa"
python enroll_faces.py --name "Trace_Capel"
python enroll_faces.py --name "Andy_Pham"
# ... repeat for each student
```

### Step 2: Run Attendance Tracking

#### Basic Usage (Video File)

```bash
python main.py --source data/example1.mp4
```

This will:
- Process the video
- Track and identify students
- Save annotated video to `output/example1_annotated.mp4`
- Generate attendance reports as `example1_annotated.txt` and `example1_annotated.json`

#### Camera Mode (Live)

```bash
python main.py --source 0 --display
```

- `--source 0`: Default camera (use 1, 2, etc. for other cameras)
- `--display`: Show live video window (`-d` also works)

#### Advanced Examples

**Custom thresholds and weights**:
```bash
python main.py \
    --source data/example1.mp4 \
    --face-confidence 0.7 \
    --appearance-weight 0.5 \  # Change from 60-40 weighting to something else (automatically normalizes to add to 1) 
    --face-weight 0.5 \  
    --attendance-threshold 0.9 \  # Stricter attendance requirements
    --verbose
```

**High-quality output with display**:
```bash
python main.py \
    --source 0 \
    --display \
    --output outputs/lecture_2024_01_15.mp4 \
    --fps 30 \
    --verbose
```

**Quick test without saving video**:
```bash
python main.py \
    --source test_video.mp4 \
    --no-save \
    --display
```

---

## Command-Line Arguments

### enroll_faces.py

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--name` | str | ✓ | - | Name of person to enroll (e.g., "John_Doe") |

### main.py

#### Required Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--source`, `-s` | str | Required | Input video file path or camera index (0, 1, 2...) |

#### Output Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output`, `-o` | str | `output/[source]_annotated.mp4` | Path to output video file |
| `--no-save` | flag | False | Don't save video |

#### Display Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--display`, `-d` | flag | False | Show live video window |
| `--window-name` | str | "Classroom Monitor" | Display window title |

#### Model Paths

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--yolo-model` | str | `yolo26n.pt` | YOLO detection model path |
| `--pose-model` | str | `yolo26n-pose.pt` | YOLO pose model path |
| `--known-faces` | str | `models/known_faces.pkl` | Enrolled faces database |
| `--appearance-model` | str | `osnet_x0_75_imagenet.pth` | Appearance feature model |

#### Recognition Thresholds

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--face-confidence`, `-fc` | float | 0.6 | Face recognition threshold (0.0-1.0). Higher = stricter |

#### Hybrid Embedding Weights

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--appearance-weight` | float | 0.6 | Weight for appearance embedding (0.0-1.0) |
| `--face-weight` | float | 0.4 | Weight for face embedding (0.0-1.0) |

> **Note**: Weights should sum to 1.0. System will normalize automatically if not.

#### Video Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--fps` | float | Auto-detect | Override output video FPS |

#### Device Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--device` | str | Auto | Device: `cuda`, `cpu`, or `mps` |

#### Attendance Tracking

| Argument | Type | Default | Description                                      |
|----------|------|---------|--------------------------------------------------|
| `--student-ids-dir` | str | `StudentIDs` | Directory with student name folders              |
| `--attendance-threshold` | float | 0.8 | Min fraction of checkpoints to mark present (80%) |
| `--camera-check-interval` | int | 10 | Seconds between checks in camera mode            |
| `--checkpoint-window` | int | 15 | Frames before/after checkpoint for detection     |
| `--hand-raise-cooldown` | float | 3.0 | Min seconds between counting hand raises         |

#### Other

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--verbose`, `-v` | flag | False | Enable detailed logging |

---

## Attendance Tracking Modes

### Video Mode (Automatic 5 Checkpoints)

When processing a video file, the system:

1. **Divides video into 5 checkpoints** (evenly spaced)
   - Example: 1000-frame video → checkpoints at frames [200, 400, 600, 800, 1000]

2. **Creates detection windows** around each checkpoint
   - Default: ±15 frames
   - Frame 200 checkpoint → window is [185-215]

3. **Records student presence**
   - If student detected anywhere in window -> checkpoint PASSED
   - If student not detected in window -> checkpoint FAILED

4. **Calculates attendance**
   - Present if: (passed checkpoints / total checkpoints) >= threshold
   - Default threshold: 0.8 (80%)
   - Example: 4/5 checkpoints = 80% -> PRESENT

### Camera Mode (Interval-Based Checkpoints)

When using live camera (`--source 0`):

1. **Creates checkpoints every N seconds**
   - Default: 10 seconds (`--camera-check-interval 10`)

2. **Each checkpoint has detection window**
   - Default: 15 frames at start of interval

3. **Continuously evaluates**
   - Attendance rate updated in real-time
   - Final report generated when stream ends (Ctrl+C or 'q')

### Adjusting Sensitivity

**More Strict** (reduce false positives):
```bash
--attendance-threshold 0.9          # Need 90% of checkpoints
--checkpoint-window 10              # Smaller detection window
--face-confidence 0.7               # Higher recognition threshold
```

**More Lenient** (reduce false negatives):
```bash
--attendance-threshold 0.6          # Need only 60% of checkpoints
--checkpoint-window 30              # Larger detection window
--face-confidence 0.5               # Lower recognition threshold
```

---

## Output Files

### Annotated Video
`output/[source]_annotated.mp4`

Visual feedback showing:
- **Green boxes**: Recognized students ("Logged!")
- **Red boxes**: Tracked but unrecognized persons ("Tracking")
- **Labels**: Student name + confidence score
- **Hand raise indicator**: "HAND RAISED" text when detected

### JSON Report
`output/[source]_annotated.json`

```json
{
  "attendance_summary": {
    "total_students": 5,
    "attendance_threshold": 0.8,
    "students_present": 4,
    "students_absent": 1
  },
  "students": {
    "Andy": {
      "present": true,
      "attendance_rate": 0.8,
      "checks_detected": 4,
      "total_checks": 5,
      "hand_raises": 2
    },
    "Janvi": {
      "present": false,
      "attendance_rate": 0.0,
      "checks_detected": 0,
      "total_checks": 5,
      "hand_raises": 0
    },
    "Robert": {
      "present": true,
      "attendance_rate": 1.0,
      "checks_detected": 5,
      "total_checks": 5,
      "hand_raises": 1
    },
    "Rohan": {
      "present": true,
      "attendance_rate": 1.0,
      "checks_detected": 5,
      "total_checks": 5,
      "hand_raises": 2
    },
    "Trace": {
      "present": true,
      "attendance_rate": 0.8,
      "checks_detected": 4,
      "total_checks": 5,
      "hand_raises": 1
    }
  }
}
```

### Text Report
`output/[source]_annotated.txt`

```
============================================================
ATTENDANCE REPORT
============================================================

Total Students: 5
Present: 4
Absent: 1
Attendance Threshold: 80%

------------------------------------------------------------
PRESENT STUDENTS
------------------------------------------------------------

Andy:
  Attendance Rate: 80.0% (4/5 checks)
  Hand Raises: 2

Robert:
  Attendance Rate: 100.0% (5/5 checks)
  Hand Raises: 1

Rohan:
  Attendance Rate: 100.0% (5/5 checks)
  Hand Raises: 2

Trace:
  Attendance Rate: 80.0% (4/5 checks)
  Hand Raises: 1

------------------------------------------------------------
ABSENT STUDENTS
------------------------------------------------------------

Janvi:
  Attendance Rate: 0.0% (0/5 checks)


```

---

## Technical Details

### Face Recognition Pipeline

1. **Detection**: MTCNN detects face bounding box
2. **Alignment**: MTCNN aligns face to canonical pose
3. **Embedding**: InceptionResnetV1 generates 512D vector
4. **Matching**: Cosine similarity against enrolled database
5. **Threshold**: Accept if similarity ≥ confidence threshold

### Appearance Feature Extraction

1. **Crop**: Extract person bounding box from YOLO
2. **Resize**: Normalize to OSNet input size
3. **Embedding**: Extract 512D appearance vector
4. **Normalization**: L2 normalize the feature vector

### Hybrid Embedding Fusion

```python
hybrid = torch.cat((appearance * w_app), (face * w_face))
hybrid = normalize(hybrid)  # L2 normalization
```

Where:
- `w_app` = `--appearance-weight` (default: 0.6)
- `w_face` = `--face-weight` (default: 0.4)
- Final dimension: 1024D (512 + 512)

### Hand Raise Detection

Uses YOLO pose keypoints (17 points):
- **Detects left hand raise**: Left wrist above left shoulder
- **Detects right hand raise**: Right wrist above right shoulder
- **Cooldown mechanism**: Prevents duplicate counts within N seconds

### Tracking Strategy

1. **Primary ID**: YOLO track ID
2. **Embedding cache**: Stores last good face + appearance embeddings
3. **Fallback**: Uses cached face embedding when detection fails
4. **Uniqueness check**: Compares new embeddings against active tracks

---

## Troubleshooting

### "No face detected" during enrollment

**Causes**:
- Poor lighting
- Face too far from camera
- Extreme angle

**Solutions**:
- Move closer to camera
- Ensure face is well-lit
- Face camera directly
- Check if green box appears (indicates detection)

### Student not recognized during tracking

**Causes**:
- Not enrolled in database
- Database pictures did not sufficiently learn the student
- Face occluded or turned away
- Lighting significantly different from enrollment
- Confidence threshold too high

**Solutions**:
```bash
# Lower confidence threshold
--face-confidence 0.5

# Increase appearance weight
--appearance-weight 0.7 --face-weight 0.3

# Re-enroll with better samples
python enroll_faces.py --name "Student_Name"
```

### False absences (student present but marked absent)

**Causes**:
- Student facing away from camera frequently
- Poor tracking during checkpoints
- Threshold too high

**Solutions**:
```bash
# Lower attendance threshold
--attendance-threshold 0.6

# Increase checkpoint window
--checkpoint-window 30

# Use more lenient face matching
--face-confidence 0.5
```

### Hand raises not detected

**Causes**:
- Hand not raised high enough (wrist must be above shoulder)
- Pose keypoints not detected accurately
- Cooldown period still active

**Solutions**:
- Ensure full arm extension above head
- Check `--hand-raise-cooldown` setting
- Verify pose model is loaded correctly

### Performance issues / low FPS

**Causes**:
- CPU-only inference
- High-resolution video
- Multiple concurrent processes

**Solutions**:
```bash
# Use GPU acceleration
--device cuda

# Reduce output FPS
--fps 15

# Use lighter models (if available)

# Disable display during processing
# (remove --display flag)
```

### CUDA out of memory

**Solutions**:
```bash
# Use CPU for some models
--device cpu

# Process at lower resolution (resize video beforehand)
```

---

## Future Enhancements

- Too much computing power required (real-time is basically unusable without an incredible PC)
- Add slight data augmentation to enrollment to help recognize faces despite lighting
- Attention/engagement scoring

---

## Acknowledgments

- **YOLOv8**: Ultralytics
- **FaceNet**: David Sandberg
- **OSNet**: Kaiyang Zhou
- **Deep Person ReID**: Torchreid library
