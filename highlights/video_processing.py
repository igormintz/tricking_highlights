import sys
import cv2
import polars as pl
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt

# Define keypoint names
KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

DIFF_BW_FRAMES = 10
N_INTRO_FRAMES = 30
N_OUTRO_FRAMES = 30

def extract_keypoints(video_path: Path, output_path: Path, save_debug=False):
    logging.info("loading model")
    model = YOLO('yolov8n-pose.pt')
    all_keypoints = []
    logging.info("opening video")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        logging.info("can't determine FPS. defaulting to 60")
        fps = 60
    logging.info(f"FPS: {fps}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    logging.info("processing video (iterating over frames)")
    for frame_number in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        results = model(frame, verbose=False)
        
        if results[0].keypoints is None:
            continue
        
        keypoints_xyn = results[0].keypoints.xyn
        keypoints_conf = results[0].keypoints.conf
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
    if save_debug:
        df.write_parquet(output_path / "raw_keypoints_data.parquet")
    return df, frames, fps

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
        if len(segment_frames) == 0: #TODO: investigate these cases.
            print(f"Warning: No frames found for highlight {i+1}")
            continue
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

def overlay_keypoints_on_frames(df, frames, fps, output_path):
    logging.info("Overlaying keypoints on frames")
    
    # Get video dimensions
    height, width = frames[0].shape[:2]
    
    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path / "keypoints_overlay.mp4"), fourcc, fps, (width, height))
    
    # Define colors for each person
    colors = plt.cm.rainbow(np.linspace(0, 1, 10))  # Up to 10 different colors
    colors = (colors[:, :3] * 255).astype(int)
    
    # Define connections for skeleton
    skeleton = [
        # Body
        ("Left Shoulder", "Right Shoulder"),
        ("Left Shoulder", "Left Elbow"),
        ("Right Shoulder", "Right Elbow"),
        ("Left Elbow", "Left Wrist"),
        ("Right Elbow", "Right Wrist"),
        ("Left Shoulder", "Left Hip"),
        ("Right Shoulder", "Right Hip"),
        ("Left Hip", "Right Hip"),
        ("Left Hip", "Left Knee"),
        ("Right Hip", "Right Knee"),
        ("Left Knee", "Left Ankle"),
        ("Right Knee", "Right Ankle"),
        # Face
        ("Left Eye", "Right Eye"),
        ("Left Eye", "Nose"),
        ("Right Eye", "Nose"),
        ("Left Eye", "Left Ear"),
        ("Right Eye", "Right Ear"),
        ("Nose", "Left Shoulder"),
        ("Nose", "Right Shoulder"),
    ]
    
    def get_keypoint_coord(kp):
        if len(kp) > 0 and kp['x'][0] is not None and kp['y'][0] is not None:
            x, y = kp['x'][0], kp['y'][0]
            # Check if the keypoint is at (0,0)
            if x == 0 and y == 0:
                return None
            return (int(x * width), int(y * height))
        return None

    # Process each frame
    for frame_number, frame in tqdm(enumerate(frames), total=len(frames), desc="Overlaying keypoints"):
        # Get keypoints for the current frame
        frame_keypoints = df.filter(pl.col("frame") == frame_number)
        
        # Draw keypoints and skeleton for each person
        for person in frame_keypoints["person"].unique():
            person_keypoints = frame_keypoints.filter(pl.col("person") == person)
            color = tuple(map(int, colors[person % len(colors)]))
            
            # Draw keypoints
            for row in person_keypoints.iter_rows(named=True):
                coord = get_keypoint_coord(pl.DataFrame([row]))
                if coord:
                    cv2.circle(frame, coord, 3, color, -1)
            
            # Draw skeleton
            for start, end in skeleton:
                start_point = get_keypoint_coord(person_keypoints.filter(pl.col("keypoint") == start))
                end_point = get_keypoint_coord(person_keypoints.filter(pl.col("keypoint") == end))
                
                if start_point and end_point:
                    cv2.line(frame, start_point, end_point, color, 2)
                elif start_point:
                    cv2.circle(frame, start_point, 3, color, -1)
                elif end_point:
                    cv2.circle(frame, end_point, 3, color, -1)
        
        # Write the frame to the output video
        out.write(frame)
    
    # Release the video writer
    out.release()
    logging.info(f"Keypoints overlay video saved to {output_path / 'keypoints_overlay.mp4'}")
