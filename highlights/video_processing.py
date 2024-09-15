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
DIFF_BW_FRAMES = 10
N_INTRO_FRAMES = 30
N_OUTRO_FRAMES = 30

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
    df.write_parquet(output_path/ "raw_keypoints_data.parquet")
    return df

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

def create_video_segments(highlight_frame_list: list[list[int]], video_path: Path, output_path: Path):
    """
    Create video segments from a list of highlight frames. for every element in the list, the first value is the start of the segment and the second value is the end of the segment.
    save the frames of the video segments to new video files.
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    for i in tqdm(range(total_frames), desc="Reading frames"):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    for i, group in enumerate(highlight_frame_list):
        start_frame, end_frame = max(0, group[0]), min(total_frames - 1, group[1])
        video_segment = frames[start_frame:end_frame+1]
        
        if video_segment:
            output_file = output_path / f"video_segment_{i}.mp4"
            out = cv2.VideoWriter(str(output_file), cv2.VideoWriter_fourcc(*'mp4v'), 30, (video_segment[0].shape[1], video_segment[0].shape[0]))
            for frame in video_segment:
                out.write(frame)
            out.release()
        
    print(f"Created {len(highlight_frame_list)} video segments")

# ... existing code ...

