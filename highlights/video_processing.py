import sys
import cv2
import polars as pl
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path

# Define keypoint names
KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

USE_N_FRAMES_PER_SECOND = 3

def extract_keypoints(video_path: Path, output_path: Path):
    model = YOLO('yolov8n-pose.pt')
    all_keypoints = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")
    use_nth_frame = int(fps // USE_N_FRAMES_PER_SECOND)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_number in tqdm(range(0, total_frames, use_nth_frame), desc="Processing video"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break

        print(f"Processing frame {frame_number}")
        results = model(frame)
        
        if results[0].keypoints is None:
            continue
        
        keypoints_xyn = results[0].keypoints.xyn
        keypoints_conf = results[0].keypoints.conf

        for person_idx, (keypoints_xy, keypoints_c) in enumerate(zip(keypoints_xyn, keypoints_conf)):
            for kp_idx, ((x, y), conf) in enumerate(zip(keypoints_xy, keypoints_c)):
                all_keypoints.append({
                    'frame': frame_number,
                    'person': person_idx,
                    'keypoint': KEYPOINT_NAMES[kp_idx],
                    'x': float(x),
                    'y': float(y),
                    'confidence': float(conf)
                })

    cap.release()
    df = pl.DataFrame(all_keypoints)
    df.write_csv(output_path/ "raw_keypoints_data.csv")
    return df

