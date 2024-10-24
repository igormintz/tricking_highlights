import argparse
import logging
from video_processing import extract_keypoints, create_highlight_lists, add_intro_and_outro, create_video_segments, overlay_keypoints_on_frames, get_video_properties
from keypoints_processing import process_keypoints
from pathlib import Path
import shutil

def create_output_dir(output_path: Path) -> None:
    """Create output_path folder. If exists, overwrite"""
    if output_path.exists():
        for item in output_path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    output_path.mkdir(parents=True, exist_ok=True)

def main(input_path: Path, output_path: Path, save_debug: bool):
    create_output_dir(output_path)
    logging.info("Starting video processing")
    
    logging.info("Extracting keypoints")
    fps, all_frames, width, height = get_video_properties(input_path)
    df = extract_keypoints(all_frames, output_path, save_debug)
    
    if save_debug:
        overlay_keypoints_on_frames(df, all_frames, fps, output_path)
    logging.info("Processing keypoints")
    df = process_keypoints(df, output_path, fps, save_debug)
    
    logging.info("Extracting frames from processed data")
    filtered_frames = df['frame'].to_list()
    
    logging.info("Creating highlight lists")
    highlight_frame_list = create_highlight_lists(filtered_frames, fps)
    
    logging.info("Adding intro and outro")
    highlight_frame_list = add_intro_and_outro(highlight_frame_list, last_frame=len(all_frames), fps=fps)
    
    logging.info("Creating video segments")
    create_video_segments(highlight_frame_list, all_frames, output_path, fps, width, height)
    
    logging.info("Video processing completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video keypoints")
    parser.add_argument("--input_path", required=True, help="Path to the input video file")
    parser.add_argument("--output_path", required=True, help="Path to save intermediate CSV files")
    parser.add_argument("--save_debug", action="store_true", default=False, help="Save debug information")
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    main(Path(args.input_path), Path(args.output_path), args.save_debug)
