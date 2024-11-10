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

USE_N_FRAMES_PER_SECOND = 3
SECONDS_BW_HIGHLIGHTS_THRESHOLD = 1




def get_video_properties(video_path: Path) -> tuple:
    logging.info("opening video")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        logging.info("can't determine FPS. defaulting to 60")
        fps = 60
    logging.info(f"Frames per second: {fps}")
    n_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    all_frames = []
    logging.info("processing video (iterating over frames)")
    for frame_number in tqdm(range(n_total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()
    return fps, all_frames, width, height

def extract_keypoints(all_frames: list, output_path: Path, model_speed='medium', save_debug=False):
    logging.info("loading model")
    model_map = {
        'fast': 'yolo11n-pose.pt',
        'medium': 'yolo11m-pose.pt',
    }
    model_path = model_map[model_speed]
    model = YOLO(model_path)
    all_keypoints = []
    for frame_number, frame in enumerate(tqdm(all_frames, desc="Extracting keypoints", dynamic_ncols=True)):
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

    df = pl.DataFrame(all_keypoints)
    if save_debug:
        logging.info("writing parquet file")
        df.write_parquet(output_path / "raw_keypoints_data.parquet")
    return df

def create_highlight_lists(highlight_frames: list, fps: float, threshold_seconds=SECONDS_BW_HIGHLIGHTS_THRESHOLD) -> list[list[int]]:
    """
    Create a list of highlight lists (consecutive frame numbers) from a list of highlight frames.
    A highlight list is a list of frames where the difference between consecutive frames is less than or equal to the threshold.
    """
    result = []
    start_frame = highlight_frames[0]
    end_frame = highlight_frames[0]
    frame_threshold = int(threshold_seconds*fps)
    for next_frame in highlight_frames[1:]:
        if next_frame - end_frame < frame_threshold:
            end_frame = next_frame
        else:
            if (end_frame - start_frame) / fps > 1:
            # append only >1 second videos
              result.append(list(range(start_frame, end_frame)))
            start_frame = next_frame
            end_frame = next_frame
    return result


import cv2
import numpy as np
from pathlib import Path

def save_frames_as_video(black_frames_with_skeleton, relevant_frames, output_path, fps, width, height, slowing_factor):
    """
    Combine two sets of frames into a single video, either side by side or top/bottom depending on orientation.
    """
    # Verify we have the same number of frames
    if len(black_frames_with_skeleton) != len(relevant_frames):
        logging.warning(f"Frame count mismatch: skeleton={len(black_frames_with_skeleton)}, original={len(relevant_frames)}")
        # Use the shorter length to avoid index errors
        n_frames = min(len(black_frames_with_skeleton), len(relevant_frames))
        black_frames_with_skeleton = black_frames_with_skeleton[:n_frames]
        relevant_frames = relevant_frames[:n_frames]
    
    # Adjust FPS for slowing factor
    output_fps = fps / slowing_factor
    
    is_landscape = width >= height

    if is_landscape:
        combined_height = height * 2
        combined_width = width
    else:
        combined_height = height
        combined_width = width * 2 

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(output_path / 'combined_video.mp4'),
        fourcc,
        output_fps / slowing_factor,
        (combined_width, combined_height)
    )
    
       # Combine and write frames
    for i, (frame1, frame2) in enumerate(zip(relevant_frames, black_frames_with_skeleton)):
        if is_landscape:
            combined = np.vstack((frame1, frame2))
        else:
            combined = np.hstack((frame1, frame2))
        
        out.write(combined)
    
    out.release()
    print("Video saved successfully")

    # Save first, middle and last frames as images for inspection
    debug_frames = [0, len(black_frames_with_skeleton)//2, len(black_frames_with_skeleton)-1]
    for frame_idx in debug_frames:
        debug_frame = np.hstack((black_frames_with_skeleton[frame_idx], relevant_frames[frame_idx]))
        cv2.imwrite(str(output_path / f'debug_frame_{frame_idx}.jpg'), debug_frame)
    

def get_keypoint_coord(kp, width, height):
    if len(kp) > 0 and kp['x'][0] is not None and kp['y'][0] is not None:
        x, y = kp['x'][0], kp['y'][0]
        # Check if the keypoint is at (0,0)
        if x == 0 and y == 0:
            return None
        return (int(x * width), int(y * height))
    return None

def overlay_keypoints_on_frames(df: pl.DataFrame, frames: list, width, height):
    logging.info("Overlaying keypoints on frames")
    
    # Define colors for each person in BGR format
    colors = plt.cm.rainbow(np.linspace(0, 1, 10))[:, :3] * 255  # Up to 10 different colors
    colors = colors.astype(int)  # Ensure colors are in integer format
    colors = [tuple(map(int, color[::-1])) for color in colors]  # Convert from RGB to BGR
    
    # Process each frame
    skeleton_frames = []
    for frame_number, frame in tqdm(enumerate(frames), total=len(frames), desc="Overlaying keypoints"):
        # Create a copy of the frame to draw on
        frame_copy = frame.copy()
        
        # Get keypoints for the current frame
        frame_keypoints = df.filter(pl.col("frame") == frame_number)
        # Draw keypoints and skeleton for each person
        for person in frame_keypoints["person"].unique():
            person_keypoints = frame_keypoints.filter(pl.col("person") == person)
            color = colors[int(person) % len(colors)]
            # Draw keypoints
            for row in person_keypoints.iter_rows(named=True):
                coord = get_keypoint_coord(pl.DataFrame([row]), width, height)
                if coord:
                    cv2.circle(frame_copy, coord, 3, color, -1)
            
            # Draw skeleton
            for start, end in skeleton:
                start_point = get_keypoint_coord(person_keypoints.filter(pl.col("keypoint") == start), width, height)
                end_point = get_keypoint_coord(person_keypoints.filter(pl.col("keypoint") == end), width, height)
                
                if start_point and end_point:
                    cv2.line(frame_copy, start_point, end_point, color, 2)
                elif start_point:
                    cv2.circle(frame_copy, start_point, 3, color, -1)
                elif end_point:
                    cv2.circle(frame_copy, end_point, 3, color, -1)
        
        skeleton_frames.append(frame_copy)
    
    if not skeleton_frames:
        logging.error("No frames were processed!")
        return
    return skeleton_frames
