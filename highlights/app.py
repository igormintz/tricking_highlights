import argparse
from video_processing import extract_keypoints, create_highlight_lists, add_intro_and_outro, create_video_segments
from keypoints_processing import process_keypoints
from pathlib import Path

def main(input_path, output_path):
    df = extract_keypoints(Path(input_path), Path(output_path))
    df = process_keypoints(df, Path(output_path))
    frames = df['frame'].to_list()
    highlight_frame_list = create_highlight_lists(frames)
    highlight_frame_list = add_intro_and_outro(highlight_frame_list)
    create_video_segments(highlight_frame_list, Path(input_path), Path(output_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video keypoints")
    parser.add_argument("--input_path", required=True, help="Path to the input video file")
    parser.add_argument("--output_path", required=True, help="Path to save intermediate CSV files")
    args = parser.parse_args()

    main(args.input_path, args.output_path)
