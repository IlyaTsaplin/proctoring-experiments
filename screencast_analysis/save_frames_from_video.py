"""
Utility script for creating a directory of frames from video files
"""

import multiprocessing
import pathlib

import cv2

READ_EVERY_NTH_FRAME = 30 * 30
PARSED_DATA_PATH = "./video_data/parsed_data"
RAW_DATA_PATH = "./video_data/raw_data"
VIDEO_NAMES = [f"video{i}" for i in range(1, 9)]
VIDEO_NAMES.extend([f"video_cheating{i}" for i in range(1, 3)])


def save_frames_from_video(video_name):
    path_to_parsed_data = pathlib.Path(PARSED_DATA_PATH) / video_name
    path_to_parsed_data.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(pathlib.Path(RAW_DATA_PATH) / f"{video_name}.webm"))

    cycle_counter = 0
    frame_counter = 0

    ret, frame = cap.read()
    while ret:
        cv2.imwrite(str(path_to_parsed_data / f"frame_{frame_counter}.jpg"), frame)
        cycle_counter += READ_EVERY_NTH_FRAME
        frame_counter += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, cycle_counter)

        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()


def main():
    with multiprocessing.Pool(5) as p:
        p.map(save_frames_from_video, VIDEO_NAMES)
        print("Done")


if __name__ == "__main__":
    main()
