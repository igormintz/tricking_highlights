import argparse
from video_processing import extract_keypoints
from keypoints_processing import process_keypoints
from pathlib import Path

def main():
    # Add argument parser
    parser = argparse.ArgumentParser(description="Process video keypoints")
    parser.add_argument("--input_path", required=True, help="Path to the input video file")
    parser.add_argument("--output_path", required=True, help="Path to save intermediate CSV files")
    args = parser.parse_args()

    # Use the provided input and output paths
    df = extract_keypoints(Path(args.input_path), Path(args.output_path))
    df = process_keypoints(df, Path(args.output_path))
    frames = df['frame'].to_list()
    

if __name__ == "__main__":
    main()
