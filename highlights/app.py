import argparse
import logging
from video_processing import extract_keypoints, overlay_keypoints_on_frames, get_video_properties, save_frames_as_video
from keypoints_processing import process_keypoints
from pathlib import Path
import shutil
from copy import deepcopy
import polars as pl
import numpy as np
SECONDS_BW_HIGHLIGHTS_THRESHOLD = 1
SLOWING_FACTOR = 2

def create_output_dir(output_path: Path) -> None:
    """Create output_path folder. If exists, overwrite"""
    if output_path.exists():
        for item in output_path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    output_path.mkdir(parents=True, exist_ok=True)

def fill_gaps(arr, threshold):
    result = []
    for i in range(len(arr) - 1):
        result.append(arr[i])
        # Check if the difference is less than the threshold
        if arr[i+1] - arr[i] < threshold:
            # Add missing numbers to fill the gap
            result.extend(range(arr[i] + 1, arr[i+1]))
    result.append(arr[-1])  # Add the last element
    return result

def main(input_path: Path, output_path: Path, model_speed:str, save_debug: bool):
    create_output_dir(output_path)
    logging.info("Starting video processing")
    
    logging.info("Extracting keypoints")
    fps, all_frames, width, height = get_video_properties(input_path)
    df = extract_keypoints(all_frames, output_path, model_speed, save_debug)
    if save_debug:
        logging.info("overlaying all frames and saving")
        overlayed = overlay_keypoints_on_frames(df, deepcopy(all_frames))
        # save all , fps, output_path, height, width,"keypoints_overlay_full"
    logging.info("Processing keypoints")
    processed_df = process_keypoints(deepcopy(df), output_path, fps, save_debug)
    
    logging.info("Extracting frames from processed data")
    filled_gaps = fill_gaps(processed_df['frame'].to_list(), threshold=SECONDS_BW_HIGHLIGHTS_THRESHOLD*fps)
    logging.info("Creating highlight lists")
    # highlight_frame_list = create_highlight_lists(filtered_frames, fps)
    
    # logging.info("Adding intro and outro")
    # highlight_frame_list = add_intro_and_outro(highlight_frame_list, last_frame=len(all_frames), fps=fps)
    relevant_df = df.filter(
        (pl.col('person').is_in(processed_df['person'].cast(pl.Int64)))
        & pl.col('frame').is_in(processed_df['frame'])
         )
    logging.info("Creating frames with skeleton")
    # create_video_segments(highlight_frame_list, all_frames, output_path, fps, width, height)
    blank_frames = [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(len(relevant_df))]
    black_frames_with_skeleton = overlay_keypoints_on_frames(relevant_df, blank_frames, width, height) # , output_path, fps, width, height
    relevant_frames = [all_frames[i] for i in relevant_df['frame'].to_list()]
    save_frames_as_video(black_frames_with_skeleton, relevant_frames, output_path, fps, width, height, SLOWING_FACTOR)
    logging.info("Video processing completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video keypoints")
    parser.add_argument("--input_path", required=True, help="Path to the input video file")
    parser.add_argument("--output_path", required=True, help="Path to save intermediate CSV files")
    parser.add_argument("--model", type=str, choices=['fast', 'medium'], default='medium',
                        help="Select YOLO model speed. 'fast' uses yolo11n-pose.pt, 'medium' uses yolo11m-pose.pt (default), 'slow' uses yolo11x-pose.pt")
    parser.add_argument("--save_debug", action="store_true", default=False, help="Save debug information")
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    main(Path(args.input_path), Path(args.output_path), args.model, args.save_debug)
