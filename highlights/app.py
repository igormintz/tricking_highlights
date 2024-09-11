from video_processing import extract_keypoints
from keypoints_processing import process_keypoints

def main():
    df = extract_keypoints("/home/igor/Downloads/input.mp4")
    df = process_keypoints(df)
    frames = df['frame'].to_list()
    

if __name__ == "__main__":
    main()
