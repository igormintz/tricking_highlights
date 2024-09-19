import argparse
import logging
from video_processing import extract_keypoints, create_highlight_lists, add_intro_and_outro, create_video_segments
from keypoints_processing import process_keypoints
from pathlib import Path

def main(input_path, output_path):
    logging.info("Starting video processing")
    
    logging.info("Extracting keypoints")
    df, frames = extract_keypoints(Path(input_path), Path(output_path))
    
    logging.info("Processing keypoints")
    df = process_keypoints(df, Path(output_path))
    
    logging.info("Extracting frames from processed data")
    frames = df['frame'].to_list()
    
    logging.info("Creating highlight lists")
    highlight_frame_list = create_highlight_lists(frames)
    
    logging.info("Adding intro and outro")
    highlight_frame_list = add_intro_and_outro(highlight_frame_list)
    
    logging.info("Creating video segments")
    create_video_segments(highlight_frame_list, frames, Path(output_path), Path(input_path))
    
    logging.info("Video processing completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video keypoints")
    parser.add_argument("--input_path", required=True, help="Path to the input video file")
    parser.add_argument("--output_path", required=True, help="Path to save intermediate CSV files")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    main(args.input_path, args.output_path)
