# Badminton Video Analysis Pipeline

## Overview

Analyze a recorded badminton game to extract:
- Shuttle hit heatmaps
- Player movement steps before each hit
- Hit locations on court
- Total distance run per player

---

## Pipeline

### 1. Player & Shuttle Detection

Use **YOLOv8** for player detection and tracking:

```bash
pip install ultralytics opencv-python
```
```
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model.track("game.mp4", persist=True)

For shuttle tracking, use TrackNetV2:
  Repo: github.com/alenzenx/TracknetV2
  Outputs shuttle (x, y) per frame

2. Hit Detection

import mediapipe as mp
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
# Detect arm swing peak → hit event

3. Court Homography

import cv2, numpy as np

src_pts = np.float32([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
dst_pts = np.float32([[0,0],[6.1,0],[6.1,13.4],[0,13.4]])
H, _ = cv2.findHomography(src_pts, dst_pts)

def to_real(px, py):
    pt = np.float32([[[px, py]]])
    return cv2.perspectiveTransform(pt, H)[0][0]

4. Player Position Tracking
positions = []
for result in results:
    for box in result.boxes:
        cx = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
        cy = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
        real_x, real_y = to_real(cx.item(), cy.item())
        positions.append((real_x, real_y))
        
5. Analytics
Player Position Heatmap
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

positions = np.array(positions)
plt.figure(figsize=(6, 13))
sns.kdeplot(x=positions[:,0], y=positions[:,1], fill=True, cmap="Reds", thresh=0.05)
plt.title("Player Position Heatmap")
plt.savefig("heatmap.png", dpi=150)

Hit Location Heatmap
hit_positions = [positions[i] for i in hit_frames]
sns.kdeplot(x=[p[0] for p in hit_positions],
            y=[p[1] for p in hit_positions],
            fill=True, cmap="Blues")
plt.title("Shuttle Hit Locations")

Distance Run
diffs = np.diff(positions[:, :2], axis=0)
distances = np.sqrt((diffs**2).sum(axis=1))
total_distance = distances.sum()
print(f"Total distance run: {total_distance:.1f} m")

Steps Before Each Hit
FPS = 30
for i, hit_frame in enumerate(hit_frames):
    prev_hit = hit_frames[i-1] if i > 0 else 0
    frames_between = hit_frame - prev_hit
    time_sec = frames_between / FPS
    print(f"Hit {i+1}: {frames_between} frames ({time_sec:.2f}s) at {hit_positions[i]}")
      
Recommended Stack
Task
Tool
Player detection/tracking
YOLOv8 + ByteTrack
Shuttle tracking
TrackNetV2
Pose / hit detection
MediaPipe Pose
Court coordinate mapping
OpenCV homography
Heatmaps & stats
seaborn + matplotlib + numpy

Quick Start Order
Get YOLOv8 player tracking working on your video
Add court homography to convert to real-world meters
Integrate TrackNetV2 for shuttle tracking
Build hit detector (trajectory reversal + pose)
Generate heatmaps and distance/step statistics EOF

 
```