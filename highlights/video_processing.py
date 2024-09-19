import sys
import cv2
import polars as pl
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path
import logging
import numpy as np

# Define keypoint names
KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

USE_N_FRAMES_PER_SECOND = 3
DIFF_BW_FRAMES = 10
N_INTRO_FRAMES = 30
N_OUTRO_FRAMES = 30

def extract_keypoints(video_path: Path, output_path: Path):
    logging.info("Extracting keypoints")
    logging.info("loading model")
    model = YOLO('yolov8n-pose.pt')
    all_keypoints = []
    logging.info("opening video")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")
    use_nth_frame = int(fps // USE_N_FRAMES_PER_SECOND)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    logging.info("processing video (iterating over frames)")
    for frame_number in tqdm(range(0, total_frames, use_nth_frame), desc="Processing video"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        print(f"Processing frame {frame_number}")
        results = model(frame, device='cpu')
        
        if results[0].keypoints is None:
            continue
        
        keypoints_xyn = results[0].keypoints.xyn
        keypoints_conf = results[0].keypoints.conf
        logging.info(f"processing keypoints for frame {frame_number}")
        if keypoints_xyn is not None and keypoints_conf is not None:
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
    logging.info("writing parquet file")
    df.write_parquet(output_path/ "raw_keypoints_data.parquet")
    return df, frames

def create_highlight_lists(highlight_frames, threshold=DIFF_BW_FRAMES) -> list[list[int]]:
    """
    Create a list of highlight lists from a list of highlight frames.
    A highlight list is a list of frames where the difference between consecutive frames is less than or equal to the threshold.
    """
    result = []
    current_group = [highlight_frames[0]]
    
    for num in highlight_frames[1:]:
        if num - current_group[-1] < 10:
            current_group.append(num)
        else:
            result.append(current_group)
            current_group = [num]
    
    result.append(current_group)  # Add the last group
    return result

def add_intro_and_outro(highlight_frame_list: list[list[int]]) -> list[list[int]]:
    """
    Add intro and outro to the highlight frame list.
    """
    # for every element in the list, add N_INTRO_FRAMES to the beginning and N_OUTRO_FRAMES to the end
    for i, group in enumerate(highlight_frame_list):
        highlight_frame_list[i] = [group[0] - N_INTRO_FRAMES, group[-1] + N_OUTRO_FRAMES]
    return highlight_frame_list

def create_video_segments(highlight_frame_list, frames, output_path, input_video_path):
    video_segments = [frames[start:end] for start, end in highlight_frame_list]
    
    # Open the original video file
    cap = cv2.VideoCapture(str(input_video_path))
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for i, segment_frames in enumerate(video_segments):
        output_file = output_path / f"highlight_{i+1}.mp4"
        
        print(f"Creating video segment {i+1} with {len(segment_frames)} frames")
        
        out = cv2.VideoWriter(str(output_file), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        for frame_number in segment_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret:
                out.write(frame)
            else:
                print(f"Warning: Could not read frame {frame_number}")
        
        out.release()
        print(f"Created highlight_{i+1}.mp4")
    
    cap.release()
    print(f"Created {len(video_segments)} video segments")

# ... existing code ...
