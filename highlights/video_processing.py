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

USE_N_FRAMES_PER_SECOND = 3
SECONDS_BW_HIGHLIGHTS_THRESHOLD = 1




def get_video_properties(video_path: Path) -> tuple:
    logging.info("opening video")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        logging.info("can't determine FPS. defaulting to 60")
        fps = 60
    logging.info(f"FPS: {fps}")
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

def extract_keypoints(all_frames: list, output_path: Path, save_debug=False):
    logging.info("loading model")
    model = YOLO('yolov8n-pose.pt')
    all_keypoints = []
    for frame_number, frame in enumerate(tqdm(all_frames, desc="Extracting keypoints")):
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
            if (end_frame - start_frame) / fps > 1
            # append only >1 second videos
              result.append(list(range(start_frame, end_frame)))
            start_frame = next_frame
            end_frame = next_frame
    return result

def add_intro_and_outro(highlight_frame_list: list[list[int]], last_frame:int, fps:float) -> list[list[int]]:
    """
    Add intro and outro to the highlight frame list.
    """
    extra_frames = int(SECONDS_BW_HIGHLIGHTS_THRESHOLD*fps/2) #to avoid overlapping
    updated_highlight_frame_list = []
    for frames in highlight_frame_list:
        if not frames:
            continue
        start_frame = max(0, frames[0] - extra_frames)
        end_frame = min(frames[-1] + extra_frames, last_frame) 
        updated_highlight_frame_list.append(list(range(start_frame, end_frame + 1)))
    
    return highlight_frame_list

def create_video_segments(highlight_frame_list, all_frames, output_path: Path, fps: float, width, height):
    for i, video_segment in enumerate(highlight_frame_list):
        if len(video_segment) == 0:  # TODO: investigate these cases.
            print(f"Warning: No frames found for highlight {i+1}")
            continue
        output_file = output_path / f"highlight_{i+1}.mp4"
        
        print(f"Creating video segment {i+1} with {len(video_segment)} frames")
        
        out = cv2.VideoWriter(str(output_file), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        # Write each frame in the video segment to the output video
        for frame_number in video_segment:
            out.write(all_frames[frame_number])
        
        out.release()
        print(f"Created highlight_{i+1}.mp4")
    
    print(f"Created {len(highlight_frame_list)} video segments")

def get_keypoint_coord(kp, width, height):
    if len(kp) > 0 and kp['x'][0] is not None and kp['y'][0] is not None:
        x, y = kp['x'][0], kp['y'][0]
        # Check if the keypoint is at (0,0)
        if x == 0 and y == 0:
            return None
        return (int(x * width), int(y * height))
    return None

def overlay_keypoints_on_frames(df: pl.DataFrame, frames: list, fps: float, output_path: Path, height, width):
    logging.info("Overlaying keypoints on frames")
    
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
                coord = get_keypoint_coord(pl.DataFrame([row]), width, height)
                if coord:
                    cv2.circle(frame, coord, 3, color, -1)
            
            # Draw skeleton
            for start, end in skeleton:
                start_point = get_keypoint_coord(person_keypoints.filter(pl.col("keypoint") == start), width, height)
                end_point = get_keypoint_coord(person_keypoints.filter(pl.col("keypoint") == end),  width, height)
                
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
